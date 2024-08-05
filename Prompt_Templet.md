# Prompt Templet

Here is the prompt template we used in our paper, using the NIPS 2020 dataset as an example.

## Difficulty

[ Grouped Question Text ]

The above questions are math problems for elementary school students. Please rate the difficulty of the above questions on a scale of 0-1, where a higher score indicates greater difficulty. Try to provide unique difficulty estimates to distinguish between the questions.

Please convert the above ratings into a Python dictionary format. The keys should be integers representing the question IDs, and the values should be the difficulty ratings of the questions.

## Ability Requirement

[ Grouped Question Text ]

The above questions are math problems for elementary school students. Please label the required skills for these questions. The relevant skills include -> [Number and Operations, Geometry and Spatial Sense, Measurement, Data Processing and Probability, Problem Solving and Reasoning], with corresponding IDs of [1, 2, 3, 4, 5]. Please label all relevant skills, and store the results in a Python array.

Please convert the above labels into a Python dictionary format, where the keys are integers representing the question IDs, and the values are arrays of skill requirement labels for each question.

## Time

[ Grouped Question Text ]

The above questions are math problems for elementary school students. We are conducting a study to estimate the average time it takes for students with an intermediate level of math skills to answer a series of questions. The student population we are considering has a normal distribution in terms of their problem-solving speed in math.

For each provided math problem, please consider the following factors:

The complexity of the problem-solving process required (e.g., simple arithmetic, multi-step problem solving, algebraic manipulation, geometric reasoning, etc.).
The difficulty level of the problem relative to the intermediate level of the elementary school curriculum.
The length and clarity of the problem text.
Any other relevant factors that may affect the time required to answer (e.g., the need to interpret graphs, the presence of distracting information, etc.).
Based on these considerations, please assign an estimated time for students to answer each question. The estimated time should be in the range of 0 to 100 seconds.

Please provide the estimates in the form of a Python dictionary. The keys should be integers representing the unique ID of each question, and the values should be floating-point numbers representing the estimated time (in seconds).

For example:

estimated_times = {
    1: 45.0,  # Question ID 1, estimated time 45.0 seconds
    2: 30.5,  # Question ID 2, estimated time 30.5 seconds
    ...
}
Please ensure that the estimates reflect a normal distribution of answering times across the set of questions.
