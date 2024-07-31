

---

  

# 1 概述

## 1.1 为什么要学习 scala

  

**原因**：

- Spark ——内存级大数据计算框架应用广泛
- Spark 使用 scala 编写

**特点**：

Scala 是一门以 java 虚拟机（JVM）为虚拟环境，并将**面向对象**编程和**函数式编程**的最佳特性结合在一起的**静态类型编程语言**（静态语言需要提前编译的如：Java、c、c++等，动态语言如：js）

  

**scala 源代码 (. Scala) 会被编译成 java 字节码 (. Class)**，然后运行在 JVM 之上，并可以**调用现有的 java 类库，实现两种语言的无缝衔接。**

  

- **同样运行在 JVM 上，可以与现存程序同时运行**
- **可直接使用 Java 类库。**
- **同 Java 一样静态类型。**
- **语法和 Java 类似，比 Java 更加简洁（简洁而并不是简单），表达性更强。**
- **同时支持面向对象、函数式编程。**
- **比 Java 更面向对象**

## 1.2 Scala 和 Java 之间的关系


```java
      javac编译器         java
java --------> .class ----------> run on JVM
		  scalac编译器        scala
scala -------> .class ----------> run on JVM
```

- 可直接使用 Java 类库
- 同样运行在 JVM 上，可以与现存程序同时运行



> [!important]  
> Java 是编译型语言还是解释型语言？Java 先编译在通过 JVM 解释执行，可以称为半编译半解释，一般不明确到底是什么类型  



## 1.3 Test

1. 新建文件 `HelloScala.scala`
    
    ```Scala
    object HelloScala { // HelloScala is a object, not a class, will create a 
        def main(args : Array[String]) : Unit = {
            println("hello,world!");
        }
    }
    // Unit表示没有输出
    ```
    
2. 在命令行编译为字节码后再运行
    
    ```Scala
    scalac HelloScala.scala
    scala helloScala
    ```
    
    如果编译生成两个 `.class` 字节码文件，`HelloScala.class` 和 `HelloScala$.class`。
    
    - 两个都是字节码，但是都不能通过 `java` 命令直接运行，原因是没有引入 Scala 的库，添加 `classpath` 就可以通过 java 执行 scala 编译成的字节码了
        
        ```Bash
        java -cp %SCALA_HOME%/lib/scala-library.jar; HelloScala
        ```
        
        使用 [Java Decompiler](http://java-decompiler.github.io/) 反编译字节码到 java 源文件可以看到引入 Scala 库的逻辑
        
        - scala 源文件中的 `HelloScala` 对象编译后成为了一个类，但对象本身编译后就是生成的另一个类 `HelloScala$` 类的单例对象 `HelloScala$.MODULE$`，称之为伴生对象。
        - `HelloScala$` 有一个 `main` 实例方法，`HelloScala` 类的静态方法通过这个单例对象转调这个实例方法。完成打印。
        - Scala 比 Java 更面向对象。
    
      
    
      
    

  

# 2 Idea 配置

## 2.1 基础配置

- 创建 Maven 项目，JDK 版本 18。
- 安装插件：Scala。一般默认都已经装了。
- Maven 项目默认用 Java 写，在 `main/` 目录下新建目录 `scala/`，然后将目录标记为 Source Root。
- 这样甚至可以在同一个项目中混用 Scala 和 Java 源文件，并互相调用。
- 需要能够添加 scala 源文件，右键项目，添加框架支持，配置 Scala SDK，选择，然后就可以右键添加 Scala 源文件了。
- 添加包，添加 Scala 类，选择**对象（obejct）**，编辑源码。

  

## 2.2 示例

```Scala
package VeryStarted

/*
  object :关键字，声明一个单例对象（伴生对象），全局只有这一个，与Hello word这个类相伴相生
 */
object HelloWorld {
  /*
  main 方法：从外部可以直接调用执行的方法
  def 方法名称(参数名称：参数类型）：返回值类型={ 方法体 }
   */
  def main(args: Array[String]): Unit = {
    println("hello world")
    /*
    调用java的类库
    */
    System.out.println("hello world from java")
  }

}
```

语法含义：

```Scala
object SingletonObject { 
def MethodName(ArgName: ArgType): RetType = { body }
}
```

`object` 关键字创建的伴生对象，可以理解为替代 Java 的 `static` 关键字的方式，将静态方法用单例对象的实例方法做了替代，做到了更纯粹的面向对象。  
  
  

# 3 变量和数据类型

  

## 3.1 注释

- 和 java 一样
- `//` 单行
- `/* */` 多行
- `/** */` 文档，方法或者类前面，便于 `scaladoc` 生成文档。

  

## 3.2 变量和常量

```Scala

var name [:VariableType] = value // 变量
val name [:ConstantType] = value // 常量
// [:VariableType] 不重要，可以自动推断，可以省略
```

因为 Scala 的函数式编程要素，所以一个指导意见就是**能用常量就不要用变量**。

- 声明变量时，**类型可以省略**，编译器会**自动推导**。
- **强数据类型**，**类型**经过给定或推导确定后就**不能修改，只能再次赋予相同类型的数值，不可以更改类型赋值**。
    
    ```Scala
    package VeryStarted
    
    object var_test {
      def main(args: Array[String]): Unit = {
        var a = 10
        // 这里会报错，int--> doule
        a = 1.0   //error
        a = 1000  //correct
        println(a)
      }
    }
    ```
    
- 变量和常量声明时，**必须有初始值**。
- 变量可变，**常量不可变**。
- **引用类型常量（实例化对象）**，不能改变常量指向的对象，**可以改变对象的属性**。
    
    1. 实例类型 (var/val)  
        决定了是否可以  
        **重新赋值**
    2. 无论是 var 还是 val，类的属性都是可以**访问**到的，但是**参数类型**决定了**属性**是否可以**更改**
    
    ```Scala
    class Student(name: String, var age:Int){
    	def printInfo():unit ={
    	println(name+ " " + age)
    	}
    }
    
    object test{
    	def main(args: Array[string]):Unit={
    	// 1. 如果是常量实例化
    	val bob = new student(name = "bob", age = 10)
    	// 不可以，因为bob是一个常量
    	val bob = new student(name = "Bob", age = 10)
    	// 2. 如果是变量
    	var bob = new student(name = "bob", age = 10)
    	// 可以，因为bob是一个常量
    	var bob = new student(name = "Bob", age = 10)
    	// 3. 属性变更，即使是Val类型
    	val bob = new Student(name = "bob", age = 10)
      // 有个前提，属性必须是变量类型(var),val的话只可以访问，不可以更改
      bob.printInfo()
      bob.age = 200 //因为age为var
      bob.printInfo()
      bob.name = "aa" //error，因为name为val
    	}
    }
    ```
    
- 不以 `;` 作为语句结尾，scala 编译器自动识别语句结尾。

  

## 3.3 标识符命名

Scala 对各种**变量、方法、函数**等命名时使用的字符序列成为**标识符**
![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20240731234447.png)



- **字母或下划线开头**，后跟字母数字下划线，和 C/C++/Java 一样。
- **操作符开头，且只包含 (+-*/#! 等)**，也是有效的标识符。这样用会用什么奇怪的好处吗？答案是灵活到天顶星的运算符重载。
- 用反引号’’包括的任意字符串，即使是同 39 个 Scala 关键字同名也可以。有点奇怪的用法，尚不知道为什么。
    - 关键字
        - `package import class obejct trait extends with type for`
        - `private protected abstract sealed final implicit lazy override`
        - `try catch finlly throw`
        - `if else match case do while for return yield`
        - `def var val`
        - `this super`
        - `new`
        - `true false null`
        - 其中 Java 没有的关键字：`object trait with implicit match yield def val var`

  

  

## 3.4 字符串输出

  

- 类型：`String`
- `+` 号连接，适用于**字符串拼接或者与变量拼接**
- `*` 字符串乘法，复制一个字符串多次
- `printf`**格式化**输出
- 字符串插值：`s"xxx${varname}"` 前缀 `s` 模板字符串
    - 前缀 `f` 格式化模板字符串，通过 `$` 获取变量值，`%` 后跟格式化字符串。
- 原始字符串：`raw"rawstringcontents${var}"`，不会考虑后跟的格式化字符串，会将内容包括%全部打印出来。
    
    ```Scala
    println(raw"the num is ${num}%2.2f")
    //会将%都打印出来
    ```
    
- 多行字符串：`""" """`。
- 输出：`print printf println ...`

  ![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20240731234554.png)



  

## 3.5 键盘输入

接收用户输入

- `import scala.io.StdIn`
- 基本语法
    
    - `StdIn.readLine()`
    - `StdIn.readInt()`
    - `StdIn.readShort() StdIn.readDouble`
    
      
    ![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20240731234638.png)



## 3.6 读取文件

1. 从文件中读取, 需要用到 `Source` 类
    
    `import scala.io.Source`
    
    `Source.fromFile(path)`
    
2. 数据写入文件  
    使用 Java 的类  
    `PrintWriter`
    
    `import java.io.PrintWriter`
    
      
    

- 示例代码
    
    ```Scala
    package VarandType
    
    import scala.io.Source
    import java.io.PrintWriter
    object FileIO {
        def main(args: Array[String]): Unit ={
            // read from file
            Source.fromFile("src/main/resources/FileIO.txt").foreach(print)
    
            // write to file
            // call java API to write
            val writer = new PrintWriter(new java.io.File("src/main/resources/WFile.txt"))
            writer.write("Nephren!")
            writer.close()
        }
    }
    ```
    
      
    

## 3.7 数据类型

- java 基本类型 `char byte short int long float double boolean`
- Java 引用类型（对象类型）
- java 基本类型对应包装类型：`Charater Byte Short Integer Long Float Double Boolean`
- Java 中不是纯粹的面向对象

  
  
**Scala 吸取了这一点，所有数据都是对象，都是**`**Any**`**的子类。**

![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20240731234659.png)


- `Any` 有两个子类：
    - `AnyVal` 值类型
    - `AnyRef` 引用类型。
- 数值类型都是 `AnyVal` 子类，和 Java 数值包装类型都一样，只有**整数**在 scala 中是 `Int`、字符是 `Char` 有点区别。
- `StringOps` 是 java 中 `String` 类增强，`AnyVal` 子类。
- `Unit` 对应 java 中的 `void`，`AnyVal` 子类。用于方法返回值的位置，**表示方法无返回值**，`Unit` 是一个类型，只有一个单例的对象，转成字符串打印出来为 `()`。
- `Void` 不是数据类型，只是一个**关键字**。
- `Null` 是一个类型，只有一个单例对象 `null` 就是空引用**，所有引用类型**`AnyRef` 的子类，这个类型主要用途是**与其他 JVM 语言互操作，几乎不在 Scala 代码中使用**。
- `Nothing` 所有类型的子类型，也称为底部类型。它常见的用途是**发出终止信号，例如抛出异常、程序退出或无限循环**。

  

### 2.7.1 整数类型（Byte、Short、Int、Long）

整数类型：存放整数值，都是有符号整数，标准补码表示

区别：每种类型在**内存中占有空间**大小不同

![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20240731234712.png)


- `Byte` 1 字节
- `Short` 2 字节
- `Int` 4 字节
- `Long` 8 字节
- 整数赋初值超出表示范围报错。
- 自动类型推断，整数字面值**默认类型**`Int`，长整型字面值必须加 `L` 后缀表示。
- 变量操作直接向下转换会失败，需要使用强制类型转换，`(a + 10).toByte`。

  

### 3.7.2 浮点类型（Float、Double）

- `Float` IEEE 754 32 位浮点数
- `Double` IEEE 754 64 位浮点数
- 字面值默认 `Double`

  

### 3.7.3 字符类型（Char）

字符类型可以表示单个字符, 字符串由一系列字符构成

- 同 java 的 `Character`，2 字节，UTF-16 编码的字符。
- 字符常量：`''`
- 类型 `Char`
- 转义：`\t \n \r \\ \" \'` etc

```Scala
val c1: Char = 'a'
val c2: Char = '3'
val c3: Char = '\t'
val c4: Char = '\n'
println("abc"+c4+"def")
```

字符变量底层保存 ASCII 码

```Scala
val c1: Char = 'a'
val c2: Char = '3'
val c3: Char = '\t'
val c4: Char = '\n'
println("abc"+c4+"def")

val i1: Int = c1
val i2: Int = c2 
```

  

### 3.7.4 布尔类型：Boolean

- True、 false
- 占一个字节

```Scala
val a: Boolean = true
val b: Boolean = false
```

  

### 3.7.5 Unit 类型、Null 类型和 Nothing 类型

所谓的空和变量类型大类一样，分为**空值**和**空引用，**另外还有一种类型：什么都没有

- `Unit` 无值，只有一个实例（），用于**函数返回值为空，**属于**anyval 类型**的子类。
- `Null` 只有一个实例 `null`，**空引用**。
- `Nothing` 
    - 最底层的类型，是其他所有一切 scala 类型的子类
    - 确定**没有正常的返回值（异常）**，可以用 Nothing 来指定返回值类型。好像意思是抛**异常**时返回 Nothing，不是特别懂。
- 示例
    
    ```Scala
    package VarandType
    
    object NullType {
        def main(arg : Array[String]) : Unit = {
            // Unit
            def f1(): Unit = {
                println("just nothing!")
            }
            // unit返回值为实例（）
            val a: Unit = f1()
            println(a) // ()
    
            // Null
            // val n:Int = null // error
            // 只有变量为引用类型的情况下，才可以被null赋值
    
            // nothing
            /*
             这里不用设置nothing类型，因为函数也有可能返回int类型，
             并且int类型为nothing的父类，所以设置为Int类型
             */
            def m2(n: Int):Int={
                if (n==0)
                    // 抛出异常
                    throw new NullPointerException()
                else
                    return n
            }
        }
    }
    ```
    
      
    

## 3.8 类型转换

### 3.8.1 数值类型自动转换

![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20240731234721.png)


- 多种数据类型混合运算，自动提升到精度最大的数据类型。
- 高精度赋值到低精度，直接报错。
- 除了图中的隐式类型转换，都需要强制类型转换。
- `Byte Short Char` 计算时会直接提升为 `Int`。
- `(Byte, Short)` 和 `Char` 之间不会互相自动转换
- `Boolean` 不能参与整数浮点运算，不能隐式转换为整数。
- 示例
    
    ```Scala
    package VarandType
    
    object TypeConversion {
        def main(arg : Array[String]) : Unit = {
    
            /*
            1. 混合运算
             */
            val a: Byte = 10
            var b: Long = 6522L
            // 结果直接定义最高精度的Long
            val result:Long = a + b
            // 如果就想得到一个想要的类型，则只能对高精度的b进行强转，变成低精度的int
            val result1:Int = a + b.toInt
    
            /*
            2.赋值操作
            高精度不可以赋值给低精度的变量
             */
            val a2: Byte = 10
            //赋值给高精度，合法
            val b2: Int = a2
            // 高精度赋值给低精度，直接会报错
            val c2:Byte = b2//error
    
            /*
            3. (byte, short)与char之间不会互相自动转换
             */
            // 3.1 赋值
            val a3: Byte = 10
            val b3: Char = a3 //error, Byte==>Char
            val c3: Char = 'd'
            val d3: Byte = c3 //error,Char==>Byte
            val e3: Int = c3
            /*
            3.2 计算. 他们三者之间确实是可以互相计算的，但是必须先将结果定义为为Int类型，否则任意两者之间都不可以做计算
             */
            val a4: Byte = 10
            val b4: Short = 25
            val c4: Char = 'c'
            val result4: Short= a4 + b4 //error
            val result5: Int = a4 + b4
            val result55: Int = a4 + b4 + c4
    
    
    
            b = 10
            println(b.toChar) // A
            println('a'.toInt) // 97
            println(2.7.toInt) // 2 
            println(-2.5.toInt) // -2
    
            println("12.3".toFloat) // 12.3
            println("13.4".toDouble.toInt) // 13.4
    
            println('a' * 3)
        }
    }
    ```
    

### 3.8.2 强制类型转换

- 高精度⇒ 低精度，但是会导致精度丢失
- `toByte toInt toChar toXXXX`
- `'a'.toInt` `2.7.toInt`
- `val n: Int = (2.6+3.5).toInt`

### 3.8.3 数值类型和 string 类型间转换

- 数值与 String 的转换：`"" + n` `"100".toInt` `"12.3".toFloat` `12.3".toDouble.toInt`
- 整数强转是二进制截取，整数高精度转低精度可能会溢出，比如 `128.toByte`。

  

  

  

# 4 运算符

## 4.1 基本运算符

- 和 Java 基本相同。
- 算术运算：`+ - * / %` ，`+` 可以用于一元正号，二元加号，还可以用作字符串加法，取模也可用于浮点数。没有自增和自减语法 `++ --`。
- 关系运算：`== != < > <= >=`
- 逻辑运算：`&& || !`， `&& ||` 所有语言都支持**短路求值**，scala 也不例外。
- 赋值运算：`= += -= *= /= %=`
- 按位运算：`& | ^ ~`
- 移位运算：`<< >> >>>`，其中 `<< >>` 是有符号左移和右移，`>>>` 无符号右移。
- Scala 中所有**运算符本质都是对象的方法调用**，拥有比 C++更灵活的运算符重载。

## 4.2 自定义运算符

自定义运算符：

- Scala 中运算符即是方法，任何具有单个参数的方法都可以用作**中缀运算符**，写作中缀表达式的写法。`10.+(1)` 即是 `10 + 1`。
- 定义时将合法的运算符（只有特殊符号构成的标识符）作为函数名称即可定义。

  

## 4.3 运算符优先级

- 当一个表达式使用多个运算符时，将**根据运算符的第一个字符来评估优先级**。内置的运算符和自定义运算符都是函数，遵守同样的规则。

```Plain
(characters not shown below)
* / %
+ -
:
= !
< >
&
^
|
(all letters, $, _)
```

- 比如下面两个表示等价：

```Plain
a + b ^? c ?^ d less a ==> b | c
((a + b) ^? (c ?^ d)) less ((a ==> b) | c)
```

  

  

# 5 流程控制

## 5.1 分支控制 if-else

- 单分支
- 双分支
- 多分支

```Scala
if (condition) {
    xxx
} else if (condition) {
    xxx
} else {
    xxx
}
```

- scala 中特殊一点，`if-else` 语句也有返回值，也就是说也可以作为表达式，定义为执行的**最后一个语句**的返回值，分支的最后一句为返回值。
- 可以**强制**要求返回 `Unit` 类型，此时忽略最后一个表达式的值，得到 `()`。
- 多种返回类型的话，赋值的目标变量类型需要指定为**具体公共父类，也可以自动推断, 否则需要将每个分支的返回值都定义为一种类型**
- scala 中没有三元条件运算符，可以用 `if (a) b else c` 替代 `a ? b : c`。
- 嵌套条件同理。

```Scala
package ControlFlows

object Branches {
    def main(args: Array[String]): Unit = {
        var age = 20

        // if-else block can have a block
        val ret = if (age >= 18) {
            println("adult")
            "adult"
        } else if (age >= 14) {
            println("teenager")
            age
        } else {
            println("child")
            age
        }
 
        println(ret) // child

        // just like a ? b : c
        age = 10
        val res = if (age >= 18) "adult" else "non-adult"
        println(res)
    }
}
```

  

## 5.2 For 循环

- 范围遍历：`for(i <- 1 to 10) {}`，其中 `1 to 10` 是 `Int` 一个方法调用，返回一个 `Range`。
    - 范围 `1 to 10` 包含**右边界**
    - `1 until 10` 不包含右边界的范围，也可以直接用 `Range` 类。
- 范围步长 `1 to 10 by 2`。
- 范围也是一个集合，也可以遍历普通集合：`for(i <- collection) {}`

  

### 5.2.1 循环守卫

循环守卫：即循环保护式，或者叫条件判断式，循环守卫为 `true` 则进入循环体内部，为 `fasle` 则跳过，类似于 `continue`。

- 写法：

```Scala
for(i <- collection if condition) {
}
```

- 等价于：

```Scala
if (i <- collection) {
    if (condition) {
    }
} 
```

### 5.2.2 循环步长

循环步长：使用 `by` ,它其实是 range 的一个方法 `1 to 10` 就是 range 对象

- 步长不可以为 0
- By 不可以直接为 double，因为他是 RichInt, 除非 range 范围也是 double 类型
    
    ```Scala
    for (i <- 1 to 10 by 2){
    	
    }
    ```
    

  

### 5.2.3 嵌套循环

- 嵌套循环同理。嵌套循环可以将条件合并到一个 `for` 中：
- 标准写法：
    
    ```Scala
    for (i <- 1 to 4) {
            for (j <- 1 to 5) {
                println("i = " + i + ", j = " + j)
            }
        }
    ```
    
- 等价写法：
    
    ```Scala
    for (i <- 1 to 4; j <- 1 to 5) {
            println("i = " + i + ", j = " + j)
        }
    ```
    
- 典型例子，乘法表：
    
    ```Scala
    for (i <- 1 to 9; j <- 1 to i) {
        print(s"$j * $i = ${i * j} \t")
        if (j == i) println()
    }
    ```
    

### 5.2.4 引入变量

- 循环中的引入变量（根据第一个循环变量发生变化的），但不是循环变量：
    
    ```Scala
    for (i <- 1 to 10; j = 10 - i) {
        println("i = " + i + ", j = " + j)
    }
    ```
    
- 循环条件也可以用 `{}`，不需要分号分割 `;`
    
    - 上面的引入变量循环等价于：
    
    ```Scala
    for {
        i <- 1 to 10
        j = 10 - i
    } {
        println("i = " + i + ", j = " + j)
    }
    ```
    

  

  

  

- 循环同样有返回值，**返回值都是空**，也就是 `Unit` 实例 `()`。
- 循环中同样可以用 `yield` 返回，外面可以接住用来操作，循环暂停，执行完后再继续循环。就像 Ruby/Python。
    
    ```Scala
    val v = for (i <- 1 to 10) yield i * i // default implementation is Vector, Vector(1, 4, 9, 16, 25, 36, 49, 64, 81, 100)
    ```
    
      
    
      
    

## 5.3 while 和 do.. While

`while` 和 `do while`：

- 为了兼容 java，**不推荐使用**，结果类型是 `Unit`。
- 不可避免**需要声明变量在循环外部**，等同于**循环内部对外部变量**造成了影响，所以不推荐使用

```Scala
while (condition) {
}
do {
} while (condition)

var a = 10
while (a>=1){
println(a)
a-=1
}
```

  

## 5.4 循环中断

- Scala 内置控制结构**去掉**了 `break continue` 关键字，为了更好适应函数式编程，推荐使用函数式风格解决。
- 使用 `breakable` 结构来实现 `break continue` 功能。
- **循环守卫**可以一定程度上替代 `continue`。
- 可以用抛出异常捕获的方式退出循环，替代 `break`。
    
    ```Scala
    try {
        for (i <- 0 to 10) {
            if (i == 3)
                throw new RuntimeException
            println(i)
        }
    } catch {
        case e: Exception => // do nothing
    }
    ```
    
- 可以使用 Scala 中的 `Breaks` 类中的 `break` 方法（只是封装了异常捕获），实现异常抛出和捕获。
    
    ```Scala
    import scala.util.control.Breaks
    // 这里要用Breaks.breakable包住
    Breaks.breakable(
        for (i <- 0 to 10) {
            if (i == 3)
                Breaks.break()
            println(i)
        }
    )
    ```