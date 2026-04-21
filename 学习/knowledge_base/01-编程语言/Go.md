---
last-updated: 2026-04-17
topics:
  - Worker Pool 动态扩容缩容与优雅关闭
---

# Go

## Worker Pool 动态扩容缩容与优雅关闭

> 来源：2026-04-17 对话归档

**问题/场景：**
用户需要一个 Go 的 worker pool 实现，要求支持动态扩容和缩容，还要能优雅关闭。

**核心知识点：**
- 使用带缓冲 channel 作为任务队列
- 维护 `minWorkers` 基线 goroutine 数量
- 当任务队列堆积时自动扩容，上限为 `maxWorkers`
- 使用 `context.CancelFunc` 通知所有 worker 退出
- 使用 `sync.WaitGroup` 确保优雅关闭等待所有 worker 完成
- 通过原子操作 `atomic.AddInt32` / `atomic.LoadInt32` 安全地跟踪当前 worker 数量

**代码示例：**
```go
package workerpool

import (
	"context"
	"sync"
	"sync/atomic"
)

type Task func()

type WorkerPool struct {
	tasks      chan Task
	workers    int32 // current worker count (atomic)
	minWorkers int
	maxWorkers int
	wg         sync.WaitGroup
	ctx        context.Context
	cancel     context.CancelFunc
	mu         sync.Mutex
}

func New(minWorkers, maxWorkers, queueSize int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	wp := &WorkerPool{
		tasks:      make(chan Task, queueSize),
		minWorkers: minWorkers,
		maxWorkers: maxWorkers,
		ctx:        ctx,
		cancel:     cancel,
	}
	for i := 0; i < minWorkers; i++ {
		wp.addWorker()
	}
	return wp
}

func (wp *WorkerPool) addWorker() {
	if int(atomic.LoadInt32(&wp.workers)) >= wp.maxWorkers {
		return
	}
	atomic.AddInt32(&wp.workers, 1)
	wp.wg.Add(1)
	go wp.workerLoop()
}

func (wp *WorkerPool) workerLoop() {
	defer wp.wg.Done()
	for {
		select {
		case task, ok := <-wp.tasks:
			if !ok {
				atomic.AddInt32(&wp.workers, -1)
				return
			}
			task()
		case <-wp.ctx.Done():
			atomic.AddInt32(&wp.workers, -1)
			return
		}
	}
}

func (wp *WorkerPool) Submit(task Task) {
	select {
	case wp.tasks <- task:
		if len(wp.tasks) > 0 {
			wp.addWorker()
		}
	default:
		wp.tasks <- task
	}
}

func (wp *WorkerPool) Shutdown() {
	wp.cancel()
	close(wp.tasks)
	wp.wg.Wait()
}
```

> 来源：2026-04-17 对话归档

**问题/场景：**
用户询问 Go 的 Worker Pool 最佳实践。

**核心知识点：**
- Worker Pool 大小应根据 CPU 核心数和任务类型（CPU-bound vs I/O-bound）调整
- 使用有缓冲 channel 避免任务提交阻塞
- 优雅处理 panic：每个 worker 应 recover 防止单个任务崩溃影响整个 pool
- 支持动态调整：根据队列长度和任务处理时间自动扩缩容
- 提供超时机制：避免任务无限期阻塞
- 监控指标：记录任务处理时间、队列长度、worker 数量等

---
