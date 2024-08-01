
# **1 函数式编程**

不同范式对比：

- **面向过程**：**按照步骤**解决问题。
- **面向对象**：分解对象、行为、属性，通过对象关系以及行为调用解决问题。耦合低，复用性高，可维护性强。
- **函数式编程**：**面向对象和面向过程都是命令式编程**，但是函数式编程不关心具体运行过程，而是关心**数据之间的映射**。
    - 纯粹的函数式编程语言中没有变量，所有量都是常量，计算过程就是不停的**表达式求值的过程**，每一段程序都有返回值。不关心底层实现，对人来说更好理解，相对地编译器处理就比较复杂。
    - **优点**：编程效率高，函数式编程的不可变性，对于函数特定输入输出是特定的，与环境上下文等无关。函数式编程无副作用，利于并行处理，所以Scala特别利于应用于大数据处理，比如Spark，Kafka框架。

## 1.1 函数基础

### 1.1.1 函数基本语法

- **函数定义**：
    
    ```scala
    def func(arg1: TypeOfArg1, arg2: ...): RetType = {
        ...
    }
    ```
    

![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20240801222008.png)


### **1.1.2 函数和方法的区别**

- 函数
    - 为完成某一功能的程序语句的集合
    - 函数式编程语言中，函数是一等公民（可以像对象一样赋值、作为参数返回值），可以在任何代码块中定义函数。
- 方法
    - 类、对象中的函数称为方法
    - 一般将定义在类或对象中（最外层）的函数称为方法，而定义在方法中（内层）的称为函数。广义上都是函数，调用时类.function( )

### 1.1.3 函数返回值

- 返回值
    
    - 用`return`返回，不写的话会使用<span style="background:rgba(240, 107, 5, 0.2)">最后一行代码</span>作为返回值。
    - 无返回值`Unit`时
        - 可以用`return`
        - 可以用`return ()`
        - 可以不返回。
    - 其他时候只需要返回值是返回值类型的子类对象就行
    
    ```scala
    object Function {
        def main(args: Array[String]): Unit = {
            // 无参，无返回值
            def f1(): Unit = {
                println("hello")
                return ()
            }
            println(f1())
    
            //无参，有返回值
            def f2(): Int = {
                println("f2() : Int")
                return 1
            }
            println(f2())
    
            // 有参，有返回值
            def f3(name: String): Int = {
                println(s"f3(${name}): Unit")
                return 3
            }
            println(f3("alice"))
    
            // 多参，无返回值
            def f4(str:String*): Unit = {
                println(str)
                for (s <- str) {
                    print(s"$s ")
                }
                println()
            }
            f4()
            f4("alice")
            f4("aaa", "bb", "cc")
    
        }
    }
    ```
    

### 1.1.3 参数

- 函数参数：
    
    - **可变参数**，类似于Java，使用数组包装。
        
        - `def f4(str:String*): Unit = {}`。
            
        - 可变参数当做**数组**来使用，当多个可变参数时，**str是数组类型**。
            
        - 如果除了可变参数还有其他参数，需要将**可变参数放在末尾**。
            
            ```scala
            def f2(str1: string, str2: string*): Unit={
            }
            ```
            
    - 参数默认值:
        
        - `def f5(name: String = "alice"): Unit = {}`
        - 和C++一样，默认参数可以不传，**默认参数必须全部放在末尾**。
    - 带名称传参：
        
        - 调用时带名称
            
            ```scala
            def f6(name: String, age: Int = 20, loc: String = "BeiJing"): Unit = {
                println(s"name ${name}, age ${age}, location ${loc}")
            }
            f6("Bob")
            f6("Alice", loc = "Xi'An")
            f6("Michael", 30)
            ```
            
        - 不给名称的就是按顺序赋值。
            
        - 调用时带名参数必须位于实参列表末尾。
            
        - 和默认参数一起使用会很方便，比如有多个默认参数，但只想覆盖其中一个。
            
    
    ```scala
    object Function {
        def main(args: Array[String]): Unit = {
            // 可变参数，无返回值
            def f4(str:String*): Unit = {
                println(str)
                for (s <- str) {
                    print(s"$s ")
                }
                println()
            }
            f4()
            f4(str = "alice")
            f4(str = "aaa", "bb", "cc")
    
            // defualt value of parameter
            def f5(name: String = "alice"): Unit = {
                println(s"default argument, name : ${name}")
            }
            f5()
    
            // named argument
            def f6(name: String, age: Int = 20, loc: String = "BeiJing"): Unit = {
                println(s"name ${name}, age ${age}, location ${loc}")
            }
            f6("Bob")
            f6("Alice", loc = "Xi'An")
            f6("Michael", 30)
    
            // simplify paramters as you can
            def f7(name: String) = name // simplify return, {}, return type
            println(f7("alice"))
    
            def f8(name: String) { // simplify = when return type is Unit, deprecated after 2.13.0
                println(name)
            }
            def f9(): Unit = {
                println("no paramter")
            }
            f9 // deprecated after 2.13.3
    
            // anonymous function, lambda expression
            val fun = (name: String) => { println(s"$name") }
            println(fun)
            fun("Alice")
        }
    }
    ```
    

## 1.2 函数至简原则

函数至简原则：

- 能省则省。
    
- 最后一行代码会作为返回值，可以省略`return`。
    
- 函数体只有一行代码的话，可以省略花括号。
    
- 如果返回值类型能够自动推断那么可以省略。
    
- 如果函数体中用`return`做返回，那么<span style="background:#ff4d4f">返回值类型必须指定</span>。
    
- 如果声明返回`Unit`，那么函数体中使用`return`返回的值也不起作用。
    
- 如果期望是无返回值类型，那么可以省略`=`。这时候没有返回值，函数也可以叫做过程。【2.13.0已废弃，能编过不过会提示。】
    
- 无参函数如果声明时没有加`()`，调用时可以省略`()`。【如果声明时有`()`调用也可以省略，不过2.13.3废弃了。】
    
- 不关心函数名称时，函数名称和`def`也可以省略，去掉返回值类型，将`=`修改为`=>`定义为匿名函数。
    
    ```
    val fun = (name: String) => { println("name") }
    ```