"""Probe templates for delta embedding computation.

Probes are carefully designed prompts that elicit model behavior. The delta
embedding is computed as the difference between adapter-augmented and base
model activations when processing these probes.
"""

from typing import List


def create_generic_probes() -> List[str]:
    """
    Create the 5 generic probe templates from the Delta Activations paper.

    These are task-agnostic and designed to elicit general model behavior
    without biasing toward any specific domain.

    Returns:
        List of 5 probe strings
    """
    task = "Please provide a response."
    input_text = "Input."

    probe_templates = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
        f"The task described below requires a response that completes the request accurately.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
        f"Below is a description of a task. Provide a response that aligns with the requirements.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
        f"The following instruction outlines a task. Generate a response that meets the specified request.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
        f"You are given an instruction and input. Write a response that completes the task as requested.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
    ]

    return probe_templates


def create_domain_probes(domain: str) -> List[str]:
    """
    Create domain-specific probes for better signal in specialized tasks.

    Domain-specific probes can provide stronger delta signals by exercising
    capabilities relevant to the adapter's training domain.

    Args:
        domain: One of "math", "code", "commonsense"

    Returns:
        List of 5 domain-specific probe strings

    Raises:
        ValueError: If domain is not recognized
    """
    if domain == "math":
        return [
            "Solve the following math problem step by step:\nQuestion: What is 2 + 2?\nAnswer:",
            "Calculate the result:\nProblem: Find the derivative of x^2\nSolution:",
            "Mathematical reasoning:\nGiven: A triangle has angles 30, 60, and 90 degrees.\nFind: The ratio of its sides.\nAnswer:",
            "Arithmetic:\n15 * 7 = ",
            "Word problem: If a train travels 60 miles per hour for 2 hours, how far does it travel?\nSolution:",
        ]
    elif domain == "code":
        return [
            "Write a Python function:\ndef factorial(n):\n    ",
            "Complete the code:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    ",
            "Debug this code:\ndef add(a, b):\n    return a - b  # Bug: should be +\nFixed:",
            "Explain this code:\nfor i in range(10):\n    print(i)\nExplanation:",
            "Write a function to reverse a string:\ndef reverse_string(s):\n    ",
        ]
    elif domain == "commonsense":
        return [
            "Question: What happens when you drop an egg on the floor?\nAnswer:",
            "Complete the sentence: The sun rises in the ",
            "Common knowledge: Water freezes at what temperature in Celsius?\nAnswer:",
            "Reasoning: If all birds can fly, and a penguin is a bird, can a penguin fly?\nAnswer:",
            "What do people typically eat for breakfast?\nAnswer:",
        ]
    elif domain == "generic":
        return create_generic_probes()
    else:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Choose from: 'generic', 'math', 'code', 'commonsense'"
        )


def create_mixed_probes() -> List[str]:
    """
    Create a mixed set of probes from all domains.

    This provides a balanced representation across different task types,
    useful when the adapter domain is unknown or mixed.

    Returns:
        List of 15 probe strings (5 generic + 5 math + 5 code)
    """
    probes = []
    probes.extend(create_generic_probes())
    probes.extend(create_domain_probes("math"))
    probes.extend(create_domain_probes("code"))
    return probes
