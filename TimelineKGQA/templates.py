# noqa: E501
QUESTION_TEMPLATES = {
    "simple": {
        "temporal_constrained_retrieval": {
            "subject": [
                "Who {predicate} {tail_object} from {start_time} to {end_time}?",
                "{tail_object} is {predicate} by who from {start_time} to {end_time}?",
            ],
            "object": [
                "{subject} {predicate} which organisation from {start_time} to {end_time}?",
                "Which organisation is {predicate} by {subject} from {start_time} to {end_time}?",
            ],
        },
        "timeline_position_retrieval": {
            "timestamp_start": [
                "When did {subject} {predicate} {tail_object} start?",
                "At what time did {subject} start {predicate} {tail_object}?",
            ],
            "timestamp_end": [
                "When did {subject} end {predicate} {tail_object}?",
                "At what time did {subject} finish {predicate} {tail_object}?",
            ],
            "timestamp_range": [
                "From when to when did {subject} {predicate} {tail_object}?",
                "During what time {subject} {predicate} {tail_object}?",
            ],
            "duration": [
                "How long did {subject} {predicate} {tail_object}?",
                "What is the duration of {subject} {predicate} {tail_object}?",
            ],
        },
    },
    "medium": {
        "timeline_position_retrieval_temporal_constrained_retrieval": {
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
                "meets": [
                    # This means first end time = second start time
                    "When {second_event_subject} starts {second_event_predicate} {second_event_object}, who ends {first_event_predicate} {first_event_object}?",
                    "Who ends {first_event_predicate} {first_event_object} when {second_event_subject} starts {second_event_predicate} {second_event_object}?",
                ],
                "metby": [
                    # This means first start time = second end time
                    "When {second_event_subject} ends {second_event_predicate} {second_event_object}, who starts {first_event_predicate} {first_event_object}?",
                    "Who starts {first_event_predicate} {first_event_object} when {second_event_subject} ends {second_event_predicate} {second_event_object}?",
                ],
                "starts": [
                    # This means first and second start at the same time, however, the end time of the first event is before the end time of the second event
                    "Who starts {first_event_predicate} {first_event_object}, at the same time {second_event_subject} start {second_event_predicate} {second_event_object}?",
                    "At the same time {second_event_subject} start {second_event_predicate} {second_event_object}, who starts {first_event_predicate} {first_event_object}?",
                ],
                "startedby": [
                    # This means first and second start at the same time, however, the end time of the first event is after the end time of the second event
                    "Who starts {first_event_predicate} {first_event_object}, at the same time {second_event_subject} start {second_event_predicate} {second_event_object}?",
                    "At the same time {second_event_subject} start {second_event_predicate} {second_event_object}, who starts {first_event_predicate} {first_event_object}?",
                ],
                "finishes": [
                    # This means first and second finish at the same time, however, the start time of the first event is after the start time of the second event
                    "Who finishes {first_event_predicate} {first_event_object}, at the same time {second_event_subject} finish {second_event_predicate} {second_event_object}?",
                    "At the same time {second_event_subject} finish {second_event_predicate} {second_event_object}, who finishes {first_event_predicate} {first_event_object}?",
                ],
                "finishedby": [
                    # This means first and second finish at the same time, however, the start time of the first event is before the start time of the second event
                    "Who finishes {first_event_predicate} {first_event_object}, at the same time {second_event_subject} finish {second_event_predicate} {second_event_object}?",
                    "At the same time {second_event_subject} finish {second_event_predicate} {second_event_object}, who finishes {first_event_predicate} {first_event_object}?",
                ],
                "equal": [
                    # This means first and second start and end at the same time
                    "Who {first_event_predicate} {first_event_object}, at the same time {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "Who {first_event_predicate} {first_event_object} at the same start and end time {second_event_subject} start and end {second_event_predicate} {second_event_object}?",
                ],
                "duration_before": [
                    "Who {first_event_predicate} {first_event_object} {temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "duration_after": [
                    "Who {first_event_predicate} {first_event_object} {temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
            },
            "object": {
                "before": [
                    "Before {second_event_subject} {second_event_predicate} {second_event_object}, which organisation is {first_event_predicate} by {first_event_subject}?",
                    "Which organisation is {first_event_predicate}ed by {first_event_subject} before {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "after": [
                    "{first_event_subject} {first_event_predicate} which ORGANISATION after {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "Which organisation is {first_event_predicate}ed by {first_event_subject} after {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "during": [
                    "During {second_event_subject} {second_event_predicate} {second_event_object}, which organisation is {first_event_predicate}ed by {first_event_subject}?",
                    "Which organisation is {first_event_predicate}ed {first_event_subject} while {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "Which organisation is {first_event_predicate}ed {first_event_subject} during the period when {second_event_subject} {second_event_predicate} {second_event_object}?",
                    "While {second_event_subject} {second_event_predicate} {second_event_object}, which organisation is {first_event_predicate}ed {first_event_subject}?",
                    "Which organisation is {first_event_predicate}ed {first_event_subject} in the course of {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "meets": [
                    # This means first end time = second start time, this question is asking for the first object
                    "{second_event_subject} starts {second_event_predicate} {second_event_object} at the same time {first_event_subject} ends {first_event_predicate} with which Organisation?",
                ],
                "metby": [
                    # This means first start time = second end time, this question is asking for the first object
                    "{second_event_subject} ends {second_event_predicate} {second_event_object} at the same time {first_event_subject} starts {first_event_predicate} with which Organisation?",
                ],
                "starts": [
                    # This means first and second start at the same time, however, the end time of the first event is before the end time of the second event
                    "At the same time {second_event_subject} start {second_event_predicate} {second_event_object}, in which organisation {first_event_subject} starts {first_event_predicate}",
                ],
                "startedby": [
                    # This means first and second start at the same time, however, the end time of the first event is after the end time of the second event
                    "At the same time {second_event_subject} start {second_event_predicate} {second_event_object}, in which organisation {first_event_subject} starts {first_event_predicate}",
                ],
                "finishes": [
                    # This means first and second finish at the same time, however, the start time of the first event is after the start time of the second event
                    "At the same time {second_event_subject} finish {second_event_predicate} {second_event_object}, in which organisation {first_event_subject} finishes {first_event_predicate}",
                ],
                "finishedby": [
                    # This means first and second finish at the same time, however, the start time of the first event is before the start time of the second event
                    "At the same time {second_event_subject} finish {second_event_predicate} {second_event_object}, in which organisation {first_event_subject} finishes {first_event_predicate}",
                ],
                "equal": [
                    # This means first and second start and end at the same time
                    "At the same time {second_event_subject} {second_event_predicate} {second_event_object}, in which organisation {first_event_subject} {first_event_predicate}?",
                ],
                "duration_before": [
                    "{temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}, in which organisation, {first_event_subject} {first_event_predicate}?",
                ],
                "duration_after": [
                    "{temporal_relation} {first_event_subject} {first_event_predicate} {first_event_object}, in which organisation, {first_event_subject} {first_event_predicate}?",
                ],
            },
        },
        "timeline_position_retrieval_timeline_position_retrieval": {
            # because it is the time range, then we should also ask about the duration
            "relation_union_or_intersection": {
                "intersection": [
                    "From when to when, {first_event_subject} {first_event_predicate} {first_event_object}, at the same time, {second_event_subject} {second_event_predicate} {second_event_object}?"
                ],
                "union": [
                    "From when to when, {first_event_subject} {first_event_predicate} {first_event_object} or {second_event_subject} {second_event_predicate} {second_event_object}?"
                    # From when to when, bush and obama are president? this type of question, this should be able to handled by the LLM when first_event_subject and second_event_subject are the same
                ],
            },
            # we are looking for the allen temporal relation between the two events, so the question normally is asked for relation directly, or true/false question.
            # However, it is still possible to have specific format for specific relationship question, even it is true/false question.
            # We will do choice and random for now, consider this later.
            "relation_allen": {
                # "X < Y": [],
                # "X m Y": [],
                # "X o Y": [],
                # "X fi Y": [],
                # "X di Y": [],
                # "X s Y": [],
                # "X = Y": [],
                # "X si Y": [],
                # "X d Y": [],
                # "X f Y": [],
                # "X oi Y": [],
                # "X mi Y": [],
                # "X > Y": [],
                "choice": [
                    "What's the temporal relation between {first_event_subject} {first_event_predicate} {first_event_object} and {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "true_false": [
                    "Is the temporal relation between {first_event_subject} {first_event_predicate} {first_event_object} {temporal_relation} {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
            },
            "relation_duration": {
                # duration for the intersection of the two events
                "duration": [
                    "What is the duration of {first_event_subject} {first_event_predicate} {first_event_object} jointly when {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "duration_compare": [
                    "Is the duration of {first_event_subject} {first_event_predicate} {first_event_object} {temporal_relation} the duration of {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                # This is actually for union
                "sum": [
                    "How long is the total duration of {first_event_subject} {first_event_predicate} {first_event_object} and {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
                "average": [
                    "What is the average duration of {first_event_subject} {first_event_predicate} {first_event_object} and {second_event_subject} {second_event_predicate} {second_event_object}?",
                ],
            },
        },
    },
    "complex": {
        "timeline_position_retrieval*2+temporal_constrained_retrieval": {
            "subject": [
                "Who {first_event_subject} {first_event_object}, {temporal_relation_12} {second_event_subject} {second_event_predicate} {second_event_object}, {temporal_relation_13} {third_event_subject} {third_event_predicate}, {third_event_object}?",
            ],
            "object": [
                "{first_event_subject} {first_event_predicate} which organisation, {temporal_relation_12} {second_event_subject} {second_event_predicate} {second_event_object}, {temporal_relation_13} {third_event_subject} {third_event_predicate} {third_event_object}?",
            ],
        },
        "timeline_position_retrieval*3": {
            "relation_union_or_intersection": {
                "intersection": [
                    "From when to when, {first_event_subject} {first_event_predicate} {first_event_object}, at the same time, {second_event_subject} {second_event_predicate} {second_event_object}, at the same time, {third_event_subject} {third_event_predicate} {third_event_object}?"
                ],
                "union": [
                    "From when to when, {first_event_subject} {first_event_predicate} {first_event_object} or {second_event_subject} {second_event_predicate} {second_event_object} or {third_event_subject} {third_event_predicate} {third_event_object}?"
                ],
            },
            # TODO: this is not really make sense, so we ignore it here
            "relation_duration": {
                "duration": [
                    "What is the duration of {first_event_subject} {first_event_predicate} {first_event_object} jointly when {second_event_subject} {second_event_predicate} {second_event_object} and {third_event_subject} {third_event_predicate} {third_event_object}?"
                ],
                "duration_compare": [
                    "Which one is {temporal_duration_rank} longest among {first_event_subject} {first_event_predicate} {first_event_object}, {second_event_subject} {second_event_predicate} {second_event_object}, {third_event_subject} {third_event_predicate} {third_event_object}?",
                ],
                "sum": [
                    "How long is the total duration of {first_event_subject} {first_event_predicate} {first_event_object}, {second_event_subject} {second_event_predicate} {second_event_object} and {third_event_subject} {third_event_predicate} {third_event_object}?"
                ],
                "average": [
                    "What is the average duration of {first_event_subject} {first_event_predicate} {first_event_object}, {second_event_subject} {second_event_predicate} {second_event_object} and {third_event_subject} {third_event_predicate} {third_event_object}?"
                ],
            },
            "relation_ranking": {
                "rank_start_time": [
                    "{first_event_subject} is ranking what based on the start time among {first_event_subject} {first_event_predicate} {first_event_object}, {second_event_subject} {second_event_predicate} {second_event_object} and {third_event_subject} {third_event_predicate} {third_event_object}?"
                ],
                "rank_end_time": [
                    "{first_event_subject} is ranking what based on the end time among {first_event_subject} {first_event_predicate} {first_event_object}, {second_event_subject} {second_event_predicate} {second_event_object} and {third_event_subject} {third_event_predicate} {third_event_object}?"
                ],
            },
        },
    },
}
