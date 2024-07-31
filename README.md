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

---

## Temporal Question Categorisation

We will begin with question answering datasets, as they are fundamental to any progress in this field. Without datasets,
we can't do anything. They are our climbing rope, guiding us to the other side of the Great Dividing Range.

Current available datasets for the Temporal Knowledge Graph Question Answering are limited.
For example, the most latest and popular TKGQA dataset: CronQuestions, containing limited types of questions, temporal
relations, temporal granularity is only to year level.

Our real world temporal questions is way more comphrehensive than this.

We all know that we are living on top of the timeline, and it only goes forward, no way looking back.
The questions we are asking are all related to the timeline, which is totally underesimated in current TKGQA research.

If we view all the temporal questions from the timeline perspective, we have this following types of timelines:

- **Straight Homogenous(Objective)** Timeline:
    - Exact date when it happens, for example, [2023-05-01 10:00:00, 2023-05-01 10:30:00]
    - This is normally asking question about the facts, and upon the facts, we can do the analysis.
    - For example, crime analysis, historical analysis, etc.
    - Under this timeline, human will focus more on **Temporal Logic**
- **Cycle Homogenous(Objective)** Timeline:
    - Monday, First day of Month, Spring, 21st Century, etc.
    - This is normally asking question about the patterns.
    - Under this timeline, human will focus more on **Temporal Pattern**
- **Straight Homogenous(Subjective)** Timeline:
    - If you sleep during night, it will be fast for you in the 8 hours, however, if someone is working overnight,
      time will be slow for him.
    - This is normally asking question about the perception of time.
    - How is your recent life goes?
    - Depending on the person, the perception of the meaning for the "recent" will be different.
    - Under this timeline, human will focus more on **Temporal Modifier**
- **Cycle Heterogeneous(Subjective)** Timeline:
    - History has its trend, however, it takes thousands years get the whole world into industrialization.
    - And then it only takes 100 years to get the whole world into information age.
    - So the spiaral speed of the timeline is not homogenous.
    - Under this timeline, human will focus more on **Temporal Modifier** also, but more trying to understand the
      development of human society, universe, etc.

We can not handle them all in a one go, and current TKGQA research is in front of the door of the **Straight Homogenous(
Objective)** Timeline.

We will try to advance the research in this area first, and then try to extend to the other areas.

### How human brain do the temporal question answering?

#### Information Indexing via Human Brain

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

#### Information Retrieval

So when we try to retrieval the information, espeically the temporal part of the information.
Normally we have several types:

- **Timeline Retrieval**:
    - When Bush starts his term as president of US?
        - First: **General Information Retrieval**  => [(Bush, start, president of US), (Bush, term, president of US)]
        - Second: **Timeline Retrieval** => [(Bush, start, president of US, 2000, 2000),
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
- **Timeline Retrieval**: Based on general information retrieval, recover the timeline information

Extend from this, it is retrieve the information for one fact, or you can name it event/truth, etc.
If we have multiple facts, or events, or truths, etc, after the retrieval, we need to comparison: set operation,
ranking, semantic extraction, etc.

And whether the question is complex or not is depending on how much information our brain need to process, and the
different capabilities of the brain needed to process the information.

### Temporal Questions

So when we try to classify the temporal questions, especially from the **difficulty** perspective, we classify the level
of
difficulty based on how many events involved in the question.

- **Simple**: Timeline and One Event Involved
- **Medium**: Timeline and Two Events Involved
- **Complex**: Timeline and Multiple Events Involved

![timeline](./docs/imgs/timeline_categorization.jpg)

#### Simple: Timeline and One Event Involved

- Timeline Retrieval:
    - When Bush starts his term as president of US?
        - General Information Retrieval => Timeline Recovery => Answer the question
        - Question Focus can be: *Timestamp Start, Timestamp End, Duration, Timestamp Start and End*
- Temporal Constrained Retrieval:
    - In 2009, who is the president of US?
        - General Information Retrieval => Temporal Constraint Retrieval => Answer the question
        - Question Focus can be: *Subject, Object, Predicate*. Can be more complex if we want mask out more elements

#### Medium: Timeline and Two Events Involved

- Timeline Retrieval + Timeline Retrieval:
    - Is Bush president of US when 911 happen?
        - *(General Information Retrieval => Timeline Recovery)* And *(General Information Retrieval => Timeline
          Recovery)* => *Timeline Operation* => Answer the question
        - Question Focus can be:
            - A new Time Range
            - A temporal relation (Before, After, During, etc.)
            - A list of Time Range (Ranking)
            - or Comparison of Duration
        - Key ability here is: **Timeline Operation**
- Timeline Retrieval + Temporal Constrained Retrieval:
    - When Bush is president of US, who is the president of China?
        - *(General Information Retrieval => Timeline Retrieval)* => *Temporal Semantic Operation* => *Temporal
          Constraint Retrieval* => Answer the question
        - This is same as above, Question Focus can be: *Subject, Object*
        - Key ability here is: **Temporal Semantic Operation**

#### Complex: Timeline and Multiple Events Involved

In general, question focus (answer type) will only be two types when we extend from Medium Level

- Timeline Operation
- (Subject, Predicate, Object)

So if we say Complex is 3 or n events and Timeline.

- Timeline Retrieval * n
- Timeline Retrieval * (n -1) => Semantic Operation * (n - 1)? => Temporal Constrainted Retrieval

And based on the answer type, we can classify them into:

- Factual
- Temporal

Based on the temporal relations in the question, we can classify them into:

- Set Operation
- Allen Temporal Relations
- Ranking
- Duration

Based on the temporal related capabilities, we can classify them into:

- Timeline Retrieval: Retrieve the timeline information, for example a time range, or a time point
- Temporal Constrained Retrieval: Based on the temporal constraint, retrieve the information
- Timeline Arithmetic Operation: Compare time intervals, do set operation, ranking, allen temporal relations, duration,
  etc
- Temporal Semantic Operation: Given a semnatic word and a time range, operate to get another time range

#### Key ability required

- **General Information Retrieval**: Retrieve the general information from the knowledge graph based on the question
- **Temporal Constrained Retrieval**: Filter on general information retrieval, apply the temporal constraint
- **Timeline Retrieval**: Based on general information retrieval, recover the timeline information
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

