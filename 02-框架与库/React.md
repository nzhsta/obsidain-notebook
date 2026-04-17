---
last-updated: 2026-04-17
topics:
  - useEffect cleanup 执行时机
---

# React

## useEffect cleanup 执行时机

> 来源：2026-04-17 对话归档

**问题/场景：**
用户询问 React useEffect 的 cleanup 函数什么时候执行。

**核心知识点：**
- cleanup 在组件卸载时执行
- cleanup 在依赖项变化、重新执行 effect 之前执行
- 执行顺序：先执行上一个 effect 的 cleanup，再执行新的 effect
- 常用于取消订阅、清除定时器、取消网络请求等

**代码示例：**
```jsx
useEffect(() => {
  const subscription = props.source.subscribe();
  return () => {
    // cleanup 在组件卸载或依赖变化前执行
    subscription.unsubscribe();
  };
}, [props.source]);
```

---
