AMR_EXAMPLES = [
    "The boy desires the girl to believe him.",
    "The girl made an adjustment to the machine.",
    "It is obligatory that the boy not go.",
    "The boy does not have permission to go.",
    "The regulatory documents were changed.",
    "He described the mission as a failure.",
    "The boy would rather go.",
    "It’s impossible for the boy to go.",
    "Where did the girl find the boy?",
    "Whose toy did the girl find?",
    "I know the person you saw.",
    "Do you want tea or coffee?",
    "The man is a lawyer.",
    "The boy destroyed the room.",
    "The boy is responsible for the results of his work.",
    "I observed the army moving quickly.",  # 16
    "I hardly know her.",  # lookup
    "He drove west, from Houston to Austin.",
    "I drove to Indianapolis on I-65.",
    "I drove through the tunnel.",
    "The soldier hummed a tune for the girl as he walked with her to town.",  # hum.01 instead of hum.02 (as in PropBank) -- problems with both concepts
    "There is no information about the case.",  # lookup 22
    "He worked for two hours.",
    "I ate pasta with a fork.",
    "She talked to him in French.",  # ('amr:language', 'amr:name', 'French')
    "The boy sang very beautifully.",  # lookup 26
    "The mayor proposed to lower crime by hiring more police officers.",
    "Nicole went to England by train.",
    "He went to the store to buy wood for a new fence.", # lookup 29
    "The boy murmured softly to soothe the girl, because he was worried about her.",
    "The game continued despite the rain.",
    "The boy will sing if he is given money.",
    "The boy will sing unless he is given money.",
    "The torpedo struck, causing damage to the ship.", # pb:agent_or_hitter_-_animate_only!
    "The boy provided chocolate to the girl.",
    "They built the bridge in Maryland in December.",
    "The engine of the car got rusty from the rain.",
    "The boy won the race in the Olympics.",
    "We met three times.",  # 39: ('pb:meet-03', 'pb:one_party', 'amr:we')
    "We play bridge every Wednesday afternoon.", # pb:rate-entity-91 (40)
    "The girl left because the boy arrived.", # pb:entity_in_motion_/_'comer'
    "The nation defaulted in June.",
    "The man died in his house between the field and the river.",
    "The Shanghai legal system.",
    "There was shouting, and the boy left.",  # lookup 45
    "The boy arrived and was promptly killed.",  # the causal connection is lost
    "The boy arrived and left on Tuesday.",
    "The brightest boy.",  # have.degree.91 instead of :degree
    "Nine of the twenty soldiers died.",  # ('pb:include-91', 'amr:ARG1', 'amr:soldier'), ('pb:include-91', 'amr:ARG2', 'amr:soldier2') — direction of rel, weird args
    "Four of the five survivors had the disease, including three who were diagnosed.",  # duplicated info about included patients
    "Marie Skłodowska-Curie received the Nobel Prize in 1911.",  # 'amr:Marie_Skłodowska-Curie' - should be found in wiki
    "During the past 30 years, 70% of the glaciers in the Alps have retreated.",
    "20 Canadian dollars",
    "The aircraft's velocity reached three times the speed of sound.",  # 'speed of sound' is a single entity
    # ('amr:product-of', 'amr:op1', 'amr:3') ('amr:product-of', 'amr:op2', 'amr:speed') ('amr:speed', 'amr:poss', 'amr:sound')
    "Patrick Makau finished the marathon in 2 hours, 3 minutes and 38 seconds.",
    "February 40, 2012",
    "Mary was playing chess while her sister was playing with toys.",  # text disambiguation: ('p', ':time', 'p3') => ('pb:play-01', 'amr:time', 'pb:play-01')
]