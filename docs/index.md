# TimelineKGQA

A universal temporal question-answering pair generator for any temporal knowledge graph, revealing the landscape of
Temporal Knowledge Graph Question Answering beyond the Great Dividing Range of Large Language Models.

---

- [Motivation](#motivation)
- [Timelines](#timelines)
- [How human brain do the temporal question answering?](#how-human-brain-do-the-temporal-question-answering)
    - [Information Indexing](#information-indexing)
    - [Information Retrieval](#information-retrieval)
- [Temporal Questions Categorisation](#temporal-questions-categorisation)
    - [Simple: Timeline and One Event Involved](#simple-timeline-and-one-event-involved)
    - [Medium: Timeline and Two Events Involved](#medium-timeline-and-two-events-involved)
    - [Complex: Timeline and Multiple Events Involved](#complex-timeline-and-multiple-events-involved)
    - [Other perspectives](#other-perspectives)
- [TimelineKGQA Generator](#timelinekgqa-generator)
- [Temporal Question Answering Solutions](#temporal-question-answering-solutions)
    - [RAG](#rag)
    - [TKGQA Embedding](#tkgqa-embedding)
    - [Text2SQL](#text2sql)
    - [Finetuning](#finetuning)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Evaluation Results](#evaluation-results)
        - [Systematic Comparison between RAG and TKGQA Embedding](#systematic-comparison-between-rag-and-tkgqa-embedding)
        - [Hits@1 for Text2SQL](#hits1-for-text2sql)
        - [Finetuning accuracy](#finetuning-accuracy)
- [Development Setup](#development-setup)
    - [Install the package](#install-the-package)
    - [Folder Structure](#folder-structure)

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

## Timelines

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


---

## How human brain do the temporal question answering?

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

---

## Temporal Questions Categorisation

![timeline](./imgs/timeline_categorization.jpg)

So when we try to classify the temporal questions, especially from the **difficulty** perspective, we classify the level
of difficulty based on how many events involved in the question.

- **Simple**: Timeline and One Event Involved
- **Medium**: Timeline and Two Events Involved
- **Complex**: Timeline and Multiple Events Involved

### Simple: Timeline and One Event Involved

- Timeline Retrieval:
    - When Bush starts his term as president of US?
        - General Information Retrieval => Timeline Recovery => Answer the question
        - Question Focus can be: *Timestamp Start, Timestamp End, Duration, Timestamp Start and End*
- Temporal Constrained Retrieval:
    - In 2009, who is the president of US?
        - General Information Retrieval => Temporal Constraint Retrieval => Answer the question
        - Question Focus can be: *Subject, Object, Predicate*. Can be more complex if we want mask out more elements

### Medium: Timeline and Two Events Involved

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

### Complex: Timeline and Multiple Events Involved

In general, question focus (answer type) will only be two types when we extend from Medium Level

- Timeline Operation
- (Subject, Predicate, Object)

So if we say Complex is 3 or n events and Timeline.

- Timeline Retrieval * n
- Timeline Retrieval * (n -1) => Semantic Operation * (n - 1)? => Temporal Constrainted Retrieval

### Other perspectives

And based on the **Answer Type**, we can classify them into:

- Factual
- Temporal

Based on the **Temporal Relations** in the question, we can classify them into:

- Set Operation
- Allen Temporal Relations
- Ranking
- Duration

Based on the **Temporal Capabilities**, we can classify them into:

- **Temporal Constrained Retrieval**: Filter on general information retrieval, apply the temporal constraint
- **Timeline Retrieval**: Based on general information retrieval, recover the timeline information
- **Timeline Operation**: From numeric to semantic
- **Temporal Semantic Operation**: From Semantic to Numeric

To be able to answer the temporal question, we need to have the following key abilities:

- **General Information Retrieval**: Retrieve the general information from the knowledge graph based on the question,
  you can call this semantic parsing, or semantic retrieval

---

## TimelineKGQA Generator

With the above understanding, it will not be hard to programmatically generate the temporal question answering pairs for
any temporal knowledge graph, as shown in the following figure:

![tkg](./imgs/tkg.jpg)

And then we can follow the following steps to generate the question answering pairs:

- Unify the temporal knowledge graph into the above format
- Sampling the facts/events from the knowledge graph
- Generate the question answer pairs based on the facts/events
- Question paraphrasing via LLM

Generating process is like:

![generator](./imgs/temporal_generation.jpg)

### Generated Question Answering Pairs for ICEWS Actor and CronQuestion KG

| Source KG           |         | Train      | Val        | Test       | Temporal Capabilities          | Count  |
|---------------------|---------|------------|------------|------------|--------------------------------|--------|
| **ICEWS Actor**     | Simple  | 17,982     | 5,994      | 5,994      | Temporal Constrained Retrieval | 34,498 |
|                     | Medium  | 15,990     | 5,330      | 5,330      | Timeline Position Retrieval    | 79,382 |
|                     | Complex | 19,652     | 6,550      | 6,550      | Timeline Operation             | 34,894 |
|                     |         |            |            |            | Temporal Semantic Operation    | 24,508 |
| **Total**           |         | **53,624** | **17,874** | **17,874** |                                | 89,372 |
| **CronQuestion KG** | Simple  | 7,200      | 2,400      | 2,400      | Temporal Constrained Retrieval | 19,720 |
|                     | Medium  | 8,252      | 2,751      | 2,751      | Timeline Position Retrieval    | 37,720 |
|                     | Complex | 9,580      | 3,193      | 3,193      | Timeline Arithmetic Operation  | 21,966 |
|                     |         |            |            |            | Temporal Semantic Operation    | 15,720 |
| **Total**           |         | **25,032** | **8,344**  | **8,344**  |                                | 41,720 |

| Answers Detailed Types         | ICEWS Actor | CronQuestions KG |
|--------------------------------|-------------|------------------|
| Subject                        | 17,249      | 9,860            |
| Object                         | 17,249      | 9,860            |
| Timestamp Start                | 4,995       | 2,000            |
| Timestamp End                  | 4,995       | 2,000            |
| Timestamp Range                | 4,995       | 2,000            |
| Duration                       | 4,995       | 2,000            |
| Relation Duration              | 9,971       | 4,000            |
| Relation Ranking               | 4,981       | 2,000            |
| Relation Union or Intersection | 19,942      | 8,000            |

For comparison, here is the statistics for the CronQuestions dataset:

| Difficulty | Template Count | Question Count | Question Categorization | Count       |
|------------|----------------|----------------|-------------------------|-------------|
| Simple     | 158            | 177,922        | Simple.Factual          | 106,208     |
| Medium     | 165            | 90,641         | Simple.Temporal         | 71,714      |
| Complex    | 331            | 141,437        | Medium.Factual          | 90,641      |
|            |                |                | Medium.Temporal         | 0           |
|            |                |                | Complex.Factual         | 67,709      |
|            |                |                | Complex.Temporal        | 73,728      |
| **Total**  | **654**        | **410,000**    |                         | **410,000** |

---

## Temporal Question Answering Solutions

![solution](./imgs/solutions.jpg)

### RAG

The first hot spot is the *Retrieval Augmented Generation(RAG)*, which will use the large language model embedding as
the semantic index, to retrieve question relevant information from the knowledge graph, and then generate the answer
based on the retrieved information.

**Really this will be the dominant solution in the future?**

### TKGQA Embedding

On this side of the Great Dividing Range, people focused on the graph embedding ways to solve the question answering
over the knowledge graph.
However, due to the challenge from the LLMs, people are tend to ignore LLM in their research for this stream, or just
give up this area.
There is not much work released in the past years regarding this area.

**Are this way really out of dated?**

From the technical perspective, current temporal knowledge graph embedding ways will not fit with our proposed and
generated dataset, because for the complex questions, the relevant fact will be 3, and they should have no difference
between this three.
If all three hit, then the Hits@1 is True.

So we developed a contrastive learning based temporal knowledge graph embedding way to solve this problem.

### Text2SQL

The third way may never think about being a competitor in the KGQA area, but the LLMs provide this potential, as the
knowledge graph in theory is just one table with a lot of interconnections.
So generate a sql to retrieve related information will be not that hard for the LLM.

**Will it really perform well in this area?**

### Finetuning

The last way in theory should be the easiest way for application if you have QA pairs, because if you want to fine tune
the ChatGPT, they will do it for you, all you need to do is to provide the QA pairs.
However, one of the main problem is lacking of QA pairs.
Which we have solved the problem above.

**So what's the real performance of this way if you do have enough QA pairs?**

### Evaluation Metrics

- **Hits@K**: The percentage of questions where the correct answer is within the top K retrieved answers.
- **MRR**: The mean reciprocal rank of the correct answer.

The Hits@K metric, used to evaluate the accuracy of event retrieval, is defined by the following criteria in our
scenario:

```math
\text{Hits@K} = 
\begin{cases} 
  1 & \text{if } \sum_{i=0}^{nN-1}r_i = n \\
  0 & \text{otherwise},
\end{cases}
```

where $r_i$ is an indicator function described as:

```math
r_i = 
\begin{cases}
  1 & \text{if the $i$-th retrieved triplet matches an event} \\
  0 & \text{otherwise}.
\end{cases}
```

The $n$ represents the number of involved events for the question.
In this framework, $r_i$ functions as an indicator that takes the value 1 if the $i$-th retrieved triplet corresponds to
one of the designated events, and 0 otherwise. The indexing for $r_i$ begins at 0.

The Mean Reciprocal Rank (MRR) is defined as follows:

```math
\text{MRR} = \frac{1}{Q} \sum_{q=1}^Q \frac{1}{\text{rank}_q + 1}
```

where $Q$ denotes the number of queries, and $\text{rank}_q$ is defined as the position of the first relevant document,
i.e., $\text{rank}_q = \min { i : r_i = 1 }$.
In our scenario, the definition of $\text{rank}_q$ needs to be adjusted to accommodate multiple relevance within the
same set of results.

It is defined as:

```math
\text{rank}_q = \sum_{i=0}^{\|\mathcal{F}\|} \left\lfloor \frac{i}{n} \right\rfloor r_i
```

where $|\mathcal{F}|$ is the number of facts.

### Evaluation Results

#### Systematic Comparison between RAG and TKGQA Embedding

As these two are similar approaches, so we will evaluate them together.

| Dataset              | Model         | MRR (Overall) | MRR (Simple) | MRR (Medium) | MRR (Complex) | Hits@1 (Overall) | Hits@1 (Simple) | Hits@1 (Medium) | Hits@1 (Complex) | Hits@3 (Overall) | Hits@3 (Simple) | Hits@3 (Medium) | Hits@3 (Complex) |
|----------------------|---------------|---------------|--------------|--------------|---------------|------------------|-----------------|-----------------|------------------|------------------|-----------------|-----------------|------------------|
| **ICEWS Actor**      | RAG           | 0.365         | 0.726        | 0.274        | 0.106         | 0.265            | 0.660           | 0.128           | 0.011            | 0.391            | 0.776           | 0.331           | 0.086            |
|                      | RAG\_semantic | 0.427         | 0.794        | 0.337        | 0.162         | 0.301            | 0.723           | 0.164           | 0.022            | 0.484            | 0.852           | 0.424           | 0.195            |
|                      | TimelineKGQA  | **0.660**     | **0.861**    | **0.632**    | **0.497**     | **0.486**        | **0.782**       | **0.435**       | **0.257**        | **0.858**        | **0.929**       | **0.845**       | **0.805**        |
| **CronQuestions KG** | RAG           | 0.331         | 0.771        | 0.218        | 0.101         | 0.235            | 0.704           | 0.092           | 0.009            | 0.348            | 0.824           | 0.249           | 0.077            |
|                      | RAG\_semantic | 0.344         | 0.775        | 0.229        | 0.122         | 0.237            | **0.707**       | 0.094           | 0.010            | 0.371            | **0.828**       | 0.267           | 0.122            |
|                      | TimelineKGQA  | **0.522**     | **0.788**    | **0.510**    | **0.347**     | **0.319**        | 0.676           | **0.283**       | **0.103**        | **0.758**        | 0.759           | **0.667**       | **0.834**        |

#### Hits@1 for Text2SQL

| Dataset              | Model                    | Hits@1 (Overall) | Hits@1 (Simple) | Hits@1 (Medium) | Hits@1 (Complex) |
|----------------------|--------------------------|------------------|-----------------|-----------------|------------------|
| **ICEWS Actor**      | GPT3.5\_base             | 0.179            | 0.268           | 0.170           | 0.105            |
|                      | GPT3.5\_semantic         | 0.358            | 0.537           | 0.311           | 0.232            |
|                      | GPT3.5\_semantic.oneshot | 0.432            | 0.611           | 0.354           | 0.328            |
|                      | GPT4o\_semantic.oneshot  | 0.485            | 0.650           | 0.392           | **0.408**        |
|                      | TimelineKGQA             | **0.486**        | **0.782**       | **0.435**       | 0.257            |
| **CronQuestions KG** | GPT3.5\_base             | 0.158            | 0.393           | 0.079           | 0.052            |
|                      | GPT3.5\_semantic         | 0.236            | 0.573           | 0.130           | 0.076            |
|                      | GPT3.5\_semantic.oneshot | 0.281            | 0.583           | 0.179           | 0.143            |
|                      | GPT4o\_semantic.oneshot  | **0.324**        | 0.623           | 0.201           | **0.207**        |
|                      | TimelineKGQA             | 0.319            | **0.676**       | **0.283**       | 0.103            |

#### Finetuning accuracy

| Model         | Rephrased | Question_as_answer | Simple_for_medium |
|---------------|-----------|--------------------|-------------------|
| GPT-3.5-Turbo | 0.60      | 0.18               | 0.36              |
| GPT-4o-mini   | 0.62      | 0.00               | 0.25              |

---

## Development Setup

### Install the package

```bash
# cd to current directory
cd TimelineKGQA
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
python3 -m TimelineKGQA.data_loader.load_icews --mode load_data --data_name icews_dicts
# this will create the unified knowledge graph
python3 -m TimelineKGQA.data_loader.load_icews --mode actor_unified_kg

# this will generate the question answering pairs
python3 -m TimelineKGQA.generator

```

### Folder Structure

```bash
TimelineKGQA/
├── TimelineKGQA/
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

