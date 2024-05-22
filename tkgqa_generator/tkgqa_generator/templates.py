QUESTION_TEMPLATES = {
    "simple": {
        "temporal_constrainted_retrieval": {
            "subject": [
                "Who {predicate} {object} from {start_time} to {end_time}?",
                "{object} is {predicate} by who from {start_time} to {end_time}?",
            ],
            "object": [
                "{subject} {predicate} which organisation from {start_time} to {end_time}?",
                "Which organisation is {predicate} by {subject} from {start_time} to {end_time}?",
            ],
        },
        "timeline_recovery": {
            "timestamp_start": [
                "When did {subject} {predicate} {object} start?",
                "At what time did {subject} start {predicate} {object}?",
            ],
            "timestamp_end": [
                "When did {subject} end {predicate} {object}?",
                "At what time did {subject} finish {predicate} {object}?",
            ],
            "timestamp_range": [
                "From when to when did {subject} {predicate} {object}?",
                "During what time {subject} {predicate} {object}?",
            ],
            "duration": [
                "How long did {subject} {predicate} {object}?",
                "What is the duration of {subject} {predicate} {object}?",
            ],
        },
    }
}
