---
last-updated: 2026-04-17
topics:
  - Python 列表推导式
  - Python 抽象基类
---

# Python

## Python 列表推导式

> 来源：2026-04-17 对话归档

**问题/场景：**
用户询问 Python 的列表推导式是什么。

**核心知识点：**
- 列表推导式是 Python 中简洁创建列表的语法
- 基本语法：`[expression for item in iterable]`
- 支持条件过滤：`[x for x in iterable if condition]`
- 比传统的 for 循环更简洁、更高效
- 也可用于字典推导式和集合推导式

**代码示例：**
```python
# 基本列表推导式
squares = [x**2 for x in range(10)]

# 带条件过滤
evens = [x for x in range(10) if x % 2 == 0]

# 字典推导式
square_dict = {x: x**2 for x in range(5)}
```

---

## Python 抽象基类

> 来源：2026-04-17 对话归档

**问题/场景：**
用户询问 Python 中抽象类（Abstract Class）的概念和用法，并延伸到 LangChain 框架中基类的设计实践。

**核心知识点：**
- 抽象类不能直接实例化，只能被继承，用于定义接口规范
- 使用 `abc` 模块中的 `ABC` 和 `@abstractmethod` 定义抽象类和抽象方法
- 子类必须实现所有抽象方法才能实例化
- 抽象类中可以同时包含抽象方法（强制子类实现）和普通方法（提供默认实现）
- 适用场景：强制子类遵循统一接口、框架设计中的插件接口定义

**代码示例：**
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

    def sleep(self):
        print("sleeping...")

class Dog(Animal):
    def speak(self):
        print("Woof!")

dog = Dog()
dog.speak()   # Woof!
# animal = Animal()  # TypeError: 无法实例化抽象类
```

---
