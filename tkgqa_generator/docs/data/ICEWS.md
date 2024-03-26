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

#### For each year, each month, the number of events

```sql
SELECT EXTRACT(YEAR FROM CAST("Event Date" AS DATE))  AS year,
       EXTRACT(MONTH FROM CAST("Event Date" AS DATE)) AS month,
       COUNT(*)                                       AS event_count
FROM icews
GROUP BY year, month
ORDER BY year, month;
```

![Monthly Events](../imgs/sql_year_month_events_no.png)

#### Sector Distribution

```sql

SELECT "Source Sectors", COUNT(*) AS total_records
FROM icews
GROUP BY "Source Sectors"
ORDER BY total_records DESC;
```

Majority of the events are from the `null` sector, which means no source sector assigned, the number is 4315622.
A lot of events are from multiple sectors.
Ohter than that, top sections include

```csv
"General Population / Civilian / Social,Social",540124
"Government,Police",501993
"Social,General Population / Civilian / Social",481059
"Police,Government",412783
"Government,Military",353416
"Military,Government",277339
Unidentified Forces,203532
"Government,Legislative / Parliamentary",170821
"Executive,Executive Office,Government",148222
"Dissident,Protestors / Popular Opposition / Mobs",145254
"Government,Judicial",135696
"Legislative / Parliamentary,Government",133953
Parties,133178
```

#### Country event distribution

```sql
SELECT "Country", COUNT(*) AS total_records
FROM icews
GROUP BY "Country"
ORDER BY total_records DESC;
```

254 Countries/Regions, the top 10 countries are:

```csv
India,1584754
Russian Federation,1107847
United States,1027424
China,820521
United Kingdom,505387
Australia,453535
Japan,444193
Iran,429701
Occupied Palestinian Territory,369623
Iraq,367640
```

#### Convert to GPS and plot on the map

```sql
ALTER TABLE icews
    ADD COLUMN geom geometry(Point, 4326);
UPDATE icews
SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);
```

![map](../imgs/map_demo_view.png)

#### All events from Australia

```sql
SELECT *
FROM icews
WHERE "Country" = 'Australia';
```

#### One example of the event

```
SELECT *
FROM icews
WHERE "Story ID" = '57125054';
```

![event](../imgs/event_example.png)

#### Story ID distribution

```sql
SELECT cnt, COUNT(*) AS stories
FROM (
    SELECT "Story ID", COUNT(*) AS cnt
    FROM icews
    GROUP BY "Story ID"
    HAVING COUNT(*) > 1
) AS duplicate_stories
GROUP BY cnt
ORDER BY cnt;
```

![story](../imgs/distribution_of_stories.png)


#### Get count for unique source name and target name

```sql
SELECT "Source Name", COUNT(*) AS source_count
FROM icews
GROUP BY "Source Name"
ORDER BY source_count DESC;
```

```sql
SELECT "Target Name", COUNT(*) AS target_count
FROM icews
GROUP BY "Target Name"
ORDER BY target_count DESC;
```

### Probelms

The entity do not have consertive stories.

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

