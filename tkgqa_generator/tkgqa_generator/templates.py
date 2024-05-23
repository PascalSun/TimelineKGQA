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
            "subject": {
                "before": [
                    "Who/Which Organisation {first_event_predicate} {first_event_object} before {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "Before {second_event_subject} {second_event_predicate} {second_event_object}, Who/Which Organisation {first_event_predicate} {first_event_object}?",
                    "Early than {second_event_subject} {second_event_predicate} {second_event_object}, Who/Which Organisation {first_event_predicate} {first_event_object}?",
                    "Prior to {second_event_subject} {second_event_predicate} {second_event_object}, Who/Which Organisation {first_event_predicate} {first_event_object}?",
                    "Who/Which Organisation {first_event_predicate} {first_event_object} ahead of {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "Who/Which Organisation {first_event_predicate} {first_event_object} preceding {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "Who/Which Organisation {first_event_predicate} {first_event_object} earlier than {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "Who/Which Organisation {first_event_predicate} {first_event_object} in advance of {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "after": [
                    "Who {second_event_predicate} {second_event_object} {temporal_relation} {first_event_subject} {first_event_predicate} {first_event_object}?",
                    "{temporal_relation} {first_event_subject} {first_event_predicate} {first_event_object}, who {second_event_predicate} {second_event_object}?",
                ],
                "during": [
                    "Who {first_event_predicate} {first_event_object} during {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "during {second_event_subject} {second_event_predicate} {second_event_object}, who {first_event_predicate} {first_event_object}?",
                    "Who {first_event_predicate} {first_event_object} while {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "while {second_event_subject} {second_event_predicate} {second_event_object}, who {first_event_predicate} {first_event_object}?",
                    "Who {first_event_predicate} {first_event_object} in the course of {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "In the midst of {second_event_subject} {second_event_predicate} {second_event_object}, who {first_event_predicate} {first_event_object}?",
                    "Who {first_event_predicate} {first_event_object} at the same time {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "starts": [
                    "Who starts {first_event_predicate} {first_event_object}, at the same time {second_event_subject} start {second_event_predicate} {second_event_object}?",
                ],
                "finishes": [
                    "Who finishes {first_event_predicate} {first_event_object}, at the same time {second_event_subject} finish {second_event_predicate} {second_event_object}?",
                ],
                "duration_before": [
                    "Who {first_event_predicate} {first_event_object} {temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "duration_after": [
                    "Who {second_event_predicate} {second_event_object} {temporal_relation} {first_event_subject} {first_event_predicate} {first_event_object}?",
                ],
            },
            "object": {
                "before": [
                    "{temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}, which organisation {first_event_predicate} {first_event_object}?",
                    "Which organisation {first_event_predicate} {first_event_object} {temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "after": [
                    "{temporal_relation} {first_event_subject} {first_event_predicate} {first_event_object}, which organisation {second_event_predicate} {second_event_object}?",
                    "Which organisation {second_event_predicate} {second_event_object} {temporal_relation} {first_event_subject} {first_event_predicate} {first_event_object}?",
                ],
                "during": [
                    "During {second_event_subject} {second_event_predicate} {second_event_object}, which organisation is {first_event_predicate}ed {first_event_object}?",
                    "Which organisation is {first_event_predicate}ed {first_event_object} while {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "Which organisation is {first_event_predicate}ed {first_event_object} during the period when {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "While {second_event_subject} {second_event_predicate} {second_event_object}, which organisation is {first_event_predicate}ed {first_event_object}?",
                    "Which organisation is {first_event_predicate}ed {first_event_object} in the course of {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "starts": [
                    "{temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}, which organisation starts {first_event_predicate} {first_event_object}?",
                ],
                "finishes": [
                    "{temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}, which organisation finishes {first_event_predicate} {first_event_object}?",
                ],
                "duration_before": [
                    "{temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}, which organisation {first_event_predicate} {first_event_object}?",
                ],
                "duration_after": [
                    "{temporal_relation} {first_event_subject} {first_event_predicate} {first_event_object}, which organisation {second_event_predicate} {second_event_object}?",
                ],
            },
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
