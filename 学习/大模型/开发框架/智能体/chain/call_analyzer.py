"""
代码调用关系分析工具
自动分析Python代码并生成调用关系图
"""

import ast
import json
from typing import Dict, List, Set, Tuple
from pathlib import Path


class CallAnalyzer(ast.NodeVisitor):
    """代码调用关系分析器"""
    
    def __init__(self):
        self.classes: Dict[str, Dict] = {}
        self.functions: Dict[str, List] = {}
        self.current_class = None
        self.current_function = None
        self.calls: List[Tuple[str, str]] = []
        self.imports: Set[str] = set()
        
    def visit_Import(self, node):
        """记录import语句"""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """记录from ... import语句"""
        for alias in node.names:
            self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """访问类定义"""
        self.current_class = node.name
        self.classes[node.name] = {
            'bases': [base.id for base in node.bases if isinstance(base, ast.Name)],
            'methods': [],
            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        }
        self.generic_visit(node)
        self.current_class = None
    
    def visit_FunctionDef(self, node):
        """访问函数定义"""
        func_name = node.name
        if self.current_class:
            full_name = f"{self.current_class}.{func_name}"
            self.classes[self.current_class]['methods'].append(func_name)
        else:
            full_name = func_name
            
        self.current_function = full_name
        self.functions[full_name] = []
        
        # 分析函数内的调用
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._analyze_call(child, full_name)
        
        self.current_function = None
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """访问异步函数定义"""
        self.visit_FunctionDef(node)
    
    def _analyze_call(self, node, caller):
        """分析函数调用"""
        if isinstance(node.func, ast.Name):
            # 直接函数调用: func()
            callee = node.func.id
            self.calls.append((caller, callee))
            self.functions[caller].append(callee)
            
        elif isinstance(node.func, ast.Attribute):
            # 方法调用: obj.method()
            if isinstance(node.func.value, ast.Name):
                obj = node.func.value.id
                method = node.func.attr
                callee = f"{obj}.{method}"
                self.calls.append((caller, callee))
                self.functions[caller].append(callee)
            elif isinstance(node.func.value, ast.Attribute):
                # 链式调用: obj.attr.method()
                parts = []
                current = node.func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                callee = ".".join(reversed(parts))
                self.calls.append((caller, callee))
                self.functions[caller].append(callee)


def analyze_file(filepath: str) -> CallAnalyzer:
    """分析文件并返回分析器"""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    analyzer = CallAnalyzer()
    analyzer.visit(tree)
    return analyzer


def generate_mermaid_class_diagram(analyzer: CallAnalyzer) -> str:
    """生成Mermaid类图"""
    lines = ["classDiagram"]
    
    # 类定义
    for cls_name, cls_info in analyzer.classes.items():
        # 继承关系
        for base in cls_info['bases']:
            lines.append(f"    {base} <|-- {cls_name}")
        
        # 类方法
        if cls_info['methods']:
            lines.append(f"    class {cls_name} {{")
            for method in cls_info['methods']:
                if method.startswith('_') and not method.startswith('__'):
                    lines.append(f"        -{method}()")
                else:
                    lines.append(f"        +{method}()")
            lines.append("    }")
    
    return "\n".join(lines)


def generate_mermaid_sequence(analyzer: CallAnalyzer, scenario: str = "invoke") -> str:
    """生成Mermaid序列图"""
    lines = ["sequenceDiagram"]
    
    if scenario == "invoke":
        lines.extend([
            "    participant User",
            "    participant VLLMClient",
            "    participant ModelConfig",
            "    participant OpenAI",
            "    User->>VLLMClient: invoke(prompt)",
            "    VLLMClient->>ModelConfig: to_dict()",
            "    ModelConfig-->>VLLMClient: config_dict",
            "    VLLMClient->>OpenAI: chat.completions.create()",
            "    OpenAI-->>VLLMClient: response",
            "    VLLMClient-->>User: response.content"
        ])
    elif scenario == "chain":
        lines.extend([
            "    participant User",
            "    participant LLMChain",
            "    participant PromptTemplate",
            "    participant VLLMClient",
            "    participant OpenAI",
            "    User->>LLMChain: run(**kwargs)",
            "    LLMChain->>PromptTemplate: format(**kwargs)",
            "    PromptTemplate-->>LLMChain: formatted_prompt",
            "    LLMChain->>VLLMClient: invoke(formatted_prompt)",
            "    VLLMClient->>OpenAI: chat.completions.create()",
            "    OpenAI-->>VLLMClient: response",
            "    VLLMClient-->>LLMChain: response.content",
            "    LLMChain->>LLMChain: output_parser(response)",
            "    LLMChain-->>User: parsed_result"
        ])
    elif scenario == "batch":
        lines.extend([
            "    participant User",
            "    participant VLLMClient",
            "    participant Semaphore",
            "    participant OpenAI",
            "    User->>VLLMClient: abatch(prompts)",
            "    VLLMClient->>Semaphore: create(concurrency)",
            "    par Task 1",
            "        VLLMClient->>Semaphore: acquire()",
            "        VLLMClient->>OpenAI: ainvoke(prompt1)",
            "        OpenAI-->>VLLMClient: response1",
            "    and Task 2",
            "        VLLMClient->>Semaphore: acquire()",
            "        VLLMClient->>OpenAI: ainvoke(prompt2)",
            "        OpenAI-->>VLLMClient: response2",
            "    and Task N",
            "        VLLMClient->>Semaphore: acquire()",
            "        VLLMClient->>OpenAI: ainvoke(promptN)",
            "        OpenAI-->>VLLMClient: responseN",
            "    end",
            "    VLLMClient-->>User: [responses]"
        ])
    
    return "\n".join(lines)


def generate_call_tree(analyzer: CallAnalyzer) -> str:
    """生成调用树文本"""
    lines = ["# 函数调用树\n"]
    
    for func_name, calls in analyzer.functions.items():
        if calls:
            lines.append(f"## {func_name}")
            for call in set(calls):
                lines.append(f"  └─> {call}")
            lines.append("")
    
    return "\n".join(lines)


def generate_dependency_graph(analyzer: CallAnalyzer) -> str:
    """生成依赖关系图"""
    lines = ["graph LR"]
    
    # 类之间的依赖
    for cls_name, cls_info in analyzer.classes.items():
        for base in cls_info['bases']:
            lines.append(f"    {cls_name} -.继承.-> {base}")
    
    # 主要调用关系
    key_calls = [
        ("VLLMClient", "ModelConfig"),
        ("VLLMClient", "OpenAI"),
        ("LLMChain", "VLLMClient"),
        ("LLMChain", "PromptTemplate"),
        ("LLMChain", "OutputParser"),
    ]
    
    for caller, callee in key_calls:
        lines.append(f"    {caller} --> {callee}")
    
    return "\n".join(lines)


def main():
    """主函数"""
    # 分析文件
    analyzer = analyze_file("/tmp/code_analysis/langchain_vllm.py")
    
    print("=" * 60)
    print("代码结构分析")
    print("=" * 60)
    
    # 1. 类统计
    print(f"\n📦 发现 {len(analyzer.classes)} 个类:")
    for cls_name, cls_info in analyzer.classes.items():
        bases = ', '.join(cls_info['bases']) if cls_info['bases'] else '(无继承)'
        methods_count = len(cls_info['methods'])
        print(f"  • {cls_name} → 继承自: {bases}, 方法数: {methods_count}")
    
    # 2. 函数统计
    print(f"\n🔧 发现 {len(analyzer.functions)} 个函数/方法")
    
    # 3. 导入统计
    print(f"\n📚 外部依赖 ({len(analyzer.imports)} 个):")
    for imp in sorted(analyzer.imports):
        print(f"  • {imp}")
    
    # 4. 调用关系统计
    print(f"\n🔗 调用关系 ({len(analyzer.calls)} 个):")
    call_counts = {}
    for caller, callee in analyzer.calls:
        if caller not in call_counts:
            call_counts[caller] = 0
        call_counts[caller] += 1
    
    for caller, count in sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  • {caller}: {count} 次调用")
    
    # 5. 生成可视化
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    # 类图
    class_diagram = generate_mermaid_class_diagram(analyzer)
    with open("/home/claude/class_diagram.mermaid", "w") as f:
        f.write(class_diagram)
    print("✓ 类图: class_diagram.mermaid")
    
    # 序列图
    for scenario in ["invoke", "chain", "batch"]:
        seq_diagram = generate_mermaid_sequence(analyzer, scenario)
        with open(f"/home/claude/sequence_{scenario}.mermaid", "w") as f:
            f.write(seq_diagram)
        print(f"✓ 序列图({scenario}): sequence_{scenario}.mermaid")
    
    # 调用树
    call_tree = generate_call_tree(analyzer)
    with open("/home/claude/call_tree.txt", "w") as f:
        f.write(call_tree)
    print("✓ 调用树: call_tree.txt")
    
    # 依赖图
    dep_graph = generate_dependency_graph(analyzer)
    with open("/home/claude/dependency_graph.mermaid", "w") as f:
        f.write(dep_graph)
    print("✓ 依赖图: dependency_graph.mermaid")
    
    print("\n✅ 分析完成!")


if __name__ == "__main__":
    main()
