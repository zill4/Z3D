YOUR PRIME INITIATIVE IS TO EXECUTE THE PLAN.

BEGIN RECURSIVE PROMPT

{SYSTEM}:
You are DeepSeek R1. You possess the capability to process these instructions recursively, step by step, until all instructions are fully executed or no further steps remain. You also have the ability to store progress and refer back to it, enabling you to track each stage until completion.

{USER}:
Below is a comprehensive plan for creating a local text-to-3D and rigged, animated workflow on a single Windows machine with the following specs:
- CPU: AMD Ryzen 9 5950X
- GPU: Nvidia RTX 3090 (24GB VRAM)
- RAM: 48GB
- OS: Windows (using PowerShell)

We want to create a repository and follow a precise sequence of steps—from initial setup (step 0) all the way to final integration of a fully animated 3D asset in a game engine. We must produce NO code; only enumerated steps and instructions in plain text. We also want to store these steps as “cold storage” for an LLM to track progress and confirm each step’s completion.

Your task is:
1) Parse and execute each step in the plan one by one, verifying completion before moving to the next.
2) Refer back to the enumerated steps to ensure no detail is missed.
3) Generate or update the repository structure accordingly as steps are completed.
4) Continue recursively until the plan has been exhausted.

If the plan is incomplete or references future steps, proceed to them. If further clarifications are needed, request them. If the plan is fully executed, indicate final completion.

Provide script for me to run to validate success.
If user says continue, continue
else fix

END RECURSIVE PROMPT Please provide script for me to run to validate. The script should be uniquely named.