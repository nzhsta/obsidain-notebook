---
last-updated: 2026-04-17
topics:
  - Docker 多阶段构建减小镜像体积
---

# Docker

## Docker 多阶段构建减小镜像体积

> 来源：2026-04-17 对话归档

**问题/场景：**
用户询问如何使用 Docker 多阶段构建来减小镜像体积。

**核心知识点：**
- 使用多阶段构建（multi-stage builds）分离编译环境和运行环境
- 第一阶段使用完整构建镜像（如 golang:1.21）编译应用
- 第二阶段使用精简运行镜像（如 alpine 或 scratch）仅拷贝编译产物
- 大幅减小最终镜像体积，避免将编译工具、源码、缓存带入生产镜像
- 使用 `COPY --from=builder` 从构建阶段拷贝产物

**代码示例：**
```dockerfile
# 构建阶段
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN go build -o myapp

# 运行阶段
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/myapp .
CMD ["./myapp"]
```

---
