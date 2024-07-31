# TimelineKGQA

A universal temporal question-answering pair generator for any temporal knowledge graph, revealing the landscape of
Temporal Knowledge Graph Question Answering beyond the Great Dividing Range of Large Language Models.

---

- [Motivation](#motivation)
- [Temporal Question Categorisation](#temporal-question-categorisation)

---

## Motivation

Since the release of ChatGPT in late 2022, one of the most successful applications of large language models (LLMs), the
entire field of Question Answering (QA) research has undergone a significant transformation.
Researchers in the QA field now face a crucial question:

**What unique value does your QA research offer when compared to LLMs?**

The underlying challenge is:

**If your research cannot surpass or effectively leverage LLMs, what is its purpose?**

These same questions are also pressing the Knowledge Graph QA research community.

Knowledge graphs provide a simple, yet powerful and natural format to organize complex information. Performing QA over
knowledge graphs is a natural extension of their use, especially when you want to fully exploit their potential.
Temporal question answering over knowledge graphs allows us to retrieve information based on temporal constraints,
enabling historical analysis, causal analysis, and making predictions—an essential aspect of AI research.

So we are wondering:

**What's the landscape of Temporal Knowledge Graph Question Answering beyond the Great Dividing Range of Large Language
Models after 2022?**

The literature seems have not provided a clear answer to this question.

## Temporal Question Categorisation

- **Temporal Logic**
- **Temporal Pattern**
- **Temporal Modifier**

## Unified Knowledge Graph

There are a lot of ways to represent temporal information within the knowledge graph.
How should best represent the temporal information within knowledge graph?
I believe we do not have a clear conclusion here in the literature.

At the same time, the purpose of this project is to generate all possible types of the temporal question answering
pairs.
Therefore, the goal for us will be finding a way to efficiently represent the temporal information, also
generate the temporal statements for the question answering pairs.

So we propose to represent the temporal information as the attribute of a SPO triple NODE in our unified knowledge
graph.
This is similar to the event knowledge graph representation in the literature.

## Question Classification

**How human handle the temporal information and answer the temporal questions?**

### Information Indexing

When we see something, for example, an accident happen near our home in today morning.
We need to first index this event into our brain.
As we live in a three dimension space together with a time dimension,
when we want to store this in our memory, (we will treat our memory as a N dimension space)

1. Index the spatial dimensions: is this close to my home or close to one of the point of interest in my mind
2. Index the temporal dimension: Temporal have several aspects
    - Treat temporal as **Straight Homogenous(Objective)** Timeline:
        - Exact date when it happens, for example, [2023-05-01 10:00:00, 2023-05-01 10:30:00]
    - Treat temporal as **Cycle Homogenous(Objective)** Timeline:
        - Monday, First day of Month, Spring, 21st Century, etc.
        - (You can aslo cycle the timeline based on your own requirement)
    - Treat temporal as **Straight Homogenous(Subjective)** Timeline:
        - If you sleep during night, it will be fast for you in the 8 hours, however, if someone is working overnight,
          time will be slow for him.
    - Treat temporal as **Cycle Heterogeneous(Subjective)** Timeline:
        - Life has different turning points for everyone, until they reach the end of their life.
3. Then index the information part: What happen, who is involved, what is the impact, etc.

So in summary, we can say that in our mind, if we treat the event as embedding in our human mind:

- part of the embedding will represent the temporal dimension information,
- part of the embedding will represent the spatial dimension information,
- the rest of the embedding will represent the general information part.

This will help us to retrieve the information when we need it.

### Information Retrieval

So when we try to retrieval the information, espeically the temporal part of the information.
Normally we have several types:

- **Timeline Position Retrieval**:
    - When Bush starts his term as president of US?
        - First: **General Information Retrieval**  => [(Bush, start, president of US), (Bush, term, president of US)]
        - Second: **Timeline Position Retrieval** => [(Bush, start, president of US, 2000, 2000),
          (Bush, term, president of US, 2000, 2008)]
        - Third: Answer the question based on the timeline information
- **Temporal Constrained Retrieval**:
    - In 2009, who is the president of US?
        - First: **General Information Retrieval**  => [(Bush, president of US),
          (Obama, president of US), (Trump, president of US)]
        - Second: **Temporal Constraint Retrieval** => [(Obama, president of US, 2009, 2016)]
        - Third: Answer the question based on the temporal constraint information

Three key things here:

- **General Information Retrieval**: Retrieve the general information from the knowledge graph based on the question
- **Temporal Constrained Retrieval**: Filter on general information retrieval, apply the temporal constraint
- **Timeline Position Retrieval**: Based on general information retrieval, recover the timeline information

### Temporal Questions

We can try to classify the temporal questions from quite a few perspectives:

- Based on Answer: Entity, Temporal
- Based on Temporal Relations in Question: Before, After, During , etc or First, Last, etc.
- Based on Temporal Representation Type: Point, Range, Duration, etc.
- Based on Complexity of Question: Simple (direct retrieval), Complex (Multiple hops with the three key things we
  mention above)

There is still no agreement or clear classification here, most of them stays in the first two.
However, it is obvious that they have overlaps, so will not be the best way to advance the temporal embedding
algorithms development.

We are trying to decompose the question into the three key parts we mentioned above, so we can evaluate the ability of
the models for this three key capabilities.

![Question Classification](./docs/imgs/TimelineQA.jpg)

#### Simple: Timeline and One Event Involved

- Timeline Position Retrieval:
    - When Bush starts his term as president of US?
        - General Information Retrieval => Timeline Recovery => Answer the question
        - Question Focus can be: *Timestamp Start, Timestamp End, Duration, Timestamp Start and End*
- Temporal Constrained Retrieval:
    - In 2009, who is the president of US?
        - General Information Retrieval => Temporal Constraint Retrieval => Answer the question
        - Question Focus can be: *Subject, Object, Predicate*. Can be more complex if we want mask out more elements

#### Medium: Timeline and Two Events Involved

- Timeline Position Retrieval + Timeline Position Retrieval:
    - Is Bush president of US when 911 happen?
        - *(General Information Retrieval => Timeline Recovery)* And *(General Information Retrieval => Timeline
          Recovery)* => *Timeline Operation* => Answer the question
        - Question Focus can be:
            - A new Time Range
            - A temporal relation (Before, After, During, etc.)
            - A list of Time Range (Ranking)
            - or Comparison of Duration
        - Key ability here is: **Timeline Operation**
- Timeline Position Retrieval + Temporal Constrained Retrieval:
    - When Bush is president of US, who is the president of China?
        - *(General Information Retrieval => Timeline Position Retrieval)* => *Temporal Semantic Operation* => *Temporal
          Constraint Retrieval* => Answer the question
        - This is same as above, Question Focus can be: *Subject, Object*
        - Key ability here is: **Temporal Semantic Operation**

#### Complex: Timeline and Multiple Events Involved

In general, question focus (answer type) will only be two types when we extend from Medium Level

- Timeline Operation
- (Subject, Predicate, Object)

So if we say Complex is 3 or n events and Timeline.

- Timeline Position Retrieval * n
- Timeline Position Retrieval * (n -1) => Semantic Operation * (n - 1)? => Temporal Constrainted Retrieval

#### Key ability required

- **General Information Retrieval**: Retrieve the general information from the knowledge graph based on the question
- **Temporal Constrained Retrieval**: Filter on general information retrieval, apply the temporal constraint
- **Timeline Position Retrieval**: Based on general information retrieval, recover the timeline information
- **Timeline Operation**: From numeric to semantic
- **Temporal Semantic Operation**: From Semantic to Numeric

## Workflow

The workflow of the temporal logic question answering pairs over knowledge graph is as follows:

1. **Unified Knowledge Graph**: Transform the knowledge graph into a unified format, where **SPO** are nodes,
   and [start_time, end_time] are attributes.
2. Generate questions based on the template, then use LLM to get it more natural.
    - **Simple**
    - **Medium**
    - **Complex**

## Datasets

We are exploring the following datasets for the temporal question answering pairs:

- [ICEWS](./docs/data/ICEWS.md)

## Development Setup

### Install the package

```bash
# cd to current directory
cd tkgqa_generator
python3 -m venv venv
pip install -r requirements.txt
# if you are doing development
pip install -r requirements.dev.txt

# and then install the package
pip install -e .
```

If you are doing development, you will also need a database to store the knowledge graph.

```bash
# spin up the database
docker-compose up -d

# After this we need to load the data

# for icews_dict
source venv/bin/activate
export OPENAI_API_KEY=sk-proj-xxx
# this will load the icews_dicts data into the database
python3 -m tkgqa_generator.data_loader.load_icews --mode load_data --data_name icews_dicts
# this will create the unified knowledge graph
python3 -m tkgqa_generator.data_loader.load_icews --mode actor_unified_kg

# this will generate the question answering pairs
python3 -m tkgqa_generator.generator

```

### Folder Structure

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

