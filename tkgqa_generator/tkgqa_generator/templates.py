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
    },
    "medium": {
        "timeline_recovery_temporal_constrainted_retrieval": {
            "subject": [
                "Who {first_event_predicate} {first_event_object} {temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}?",
                "{first_event_subject} {first_event_predicate} {first_event_object} {temporal_relation} who {second_event_predicate} {second_event_object}?",
            ],
            "object": [
                "{first_event_subject} {first_event_predicate} which/where {temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}?",
                "Which/where {second_event_subject} {second_event_predicate} {second_event_object} {temporal_relation} {first_event_subject} {first_event_predicate} {first_event_object}?",
            ],
        },
        "timeline_recovery_timeline_recovery": {
            # because it is the time range, then we should also ask about the duration
            "relation_union_or_intersection": [
                "How long when {first_event_subject} {first_event_predicate} {first_event_object} and {second_event_subject} {second_event_predicate} {second_event_object} at the same time?",
                "From when to when {first_event_subject} {first_event_predicate} {first_event_object} and {second_event_subject} {second_event_predicate} {second_event_object} at the same time?",
            ],
            "relation_allen": [
                "When {first_event_subject} {first_event_predicate} {first_event_object}, {second_event_subject} {second_event_predicate} {second_event_object}?",
                "At what time {first_event_subject} {first_event_predicate} {first_event_object}, {second_event_subject} {second_event_predicate} {second_event_object}?",
            ],
            "relation_ordinal": [
                "What is the order of {first_event_subject} {first_event_predicate} {first_event_object} and {second_event_subject} {second_event_predicate} {second_event_object}?",
                "Which event happened first, {first_event_subject} {first_event_predicate} {first_event_object} or {second_event_subject} {second_event_predicate} {second_event_object}?",
            ],
            "relation_duration": [
                "Is {first_event_subject} {first_event_predicate} {first_event_object} longer than {second_event_subject} {second_event_predicate} {second_event_object}?",
                "Which event took longer, {first_event_subject} {first_event_predicate} {first_event_object} or {second_event_subject} {second_event_predicate} {second_event_object}?",
                "In total, how long did {first_event_subject} {first_event_predicate} {first_event_object} and {second_event_subject} {second_event_predicate} {second_event_object}?",
            ],
        },
    },
}
