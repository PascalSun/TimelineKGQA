# TKGQA Generator

We plan to generate temporal question answering pairs from knowledge graphs from three different perspectives:

- **Temporal Logic**
- **Temporal Pattern**
- **Temporal Modifier**

## Temporal Logic

![Temporal Logic](./docs/imgs/tc-logic.png)


### Workflow

![Workflow](./docs/imgs/experiment-design.png)

## Folder Structure

```bash
tkgqa_generator/
├── tkgqa_generator/
│   ├── __init__.py
│   ├── generator.py
│   ├── processor.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_generator.py
│   └── test_processor.py
├── docs/
│   └── ...
├── examples/
│   └── basic_usage.py
├── setup.py
├── requirements.txt
├── README.md
└── LICENSE
```
