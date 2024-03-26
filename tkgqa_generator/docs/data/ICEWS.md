# ICEWS

Integrated Crisis Early Warning System

Data download link: [ICEWS](https://dataverse.harvard.edu/dataverse/icews)

There are four links there, two of them are deprecated, which are:

- ICEWS Event Aggregations
    - Which want to aggregate the events into a monthly basis
- ICEWS Events of Interest Ground Truth Data Set
    - Rather than from the temporal aspect to aggregate the events
    - this is used to aggregate the events from a semantic
      aspect
    - With the latest advanced in LLM, I will say this can be done via semantic retrieval
    - ![event_of_interest](../imgs/event-of-interest.png)

Both of them were deprecated in 2015.

ICEWS Dictionaries are kind of ontology of the datasets, which is still useful.
The core event data is the ICEWS Events data, which is from this
link: [ICEWS Events](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075)

## Exploration of the data

The data spanned from 1995 to 2023 now.

### SQL Exploration

#### Total number of events

```sql
SELECT COUNT(*)
FROM icews;
-- Results: 18662558
```

#### For each year, the number of events

```sql
SELECT EXTRACT(YEAR FROM CAST("Event Date" AS DATE)) AS year, COUNT(*) AS total_records
FROM icews
GROUP BY EXTRACT(YEAR FROM CAST("Event Date" AS DATE))
ORDER BY year;
```

![Yearly Events](../imgs/sql_year_events_no.png)

## Download data

Data is available within the link above, or this
link: [ICEWS Events](https://pascalsun.sg4.quickconnect.to/d/s/xkoI2xvSvWopVqbmdNQ0wQxh5JwknS8K/FByRCtROPpmYWCpleIfh_LAL1wc6Lysb-grMgdjDqOws)

The file we focus on will be `ICEWS Coded Event Data`

Put them into the folder `data/ICEWS`

The structure of the data is as follows:

```bash
├── data/
│   └── icews/
│      │── ICEWS Coded Event Data/
│      └── ICEWS Dictionaries/
```

