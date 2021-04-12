# STA4273 Project Repository

Final code and report for the [STA4273 course](https://www.cs.toronto.edu/~cmaddis/courses/sta4273_w21/) at the University of Toronto, taught by [Chris Maddison](https://www.cs.toronto.edu/~cmaddis/). [Blair Bilodeau](https://github.com/blairbilodeau) and I have worked on our project, **Methods for Adding Explicit Uncertainty to Deep Q-Learning**, in a private repository. This repository contains only the cleaned version of our code and final version of our report.

## Abstract

A successful reinforcement learning RL player must generalize its own experiences while considering long-term consequences as well as adequately explore the environment. While existing work has provided methods to capture the necessary uncertainty for exploration using Bayesian methods, there is not a principled method to do so with arbitrary prior mechanisms.In this work, we formalize a framework for adding explicit uncertainty into deep RL algorithms, highlight potential shortcomings of an existing ensemble algorithm (BootDQN+prior), and propose improvements inspired by both statistical optimization and the sequential decision making literature. We demonstrate that our proposed algorithms obtain smaller regret on classic benchmark RL problems.

### Author Contributions

In this project, Blair implemented the Exp3 method, implemented the GaussianParamNet objects, created a wrapper function to easily use any algorithm, wrote Sections 2 and 3, and assisted with writing the remaining sections. Also, Anthony implemented the policy gradient method, implemented the NoisyNet objects, cleaned the code in Python files, ran the final experiments and wrote Section 5, and assisted with writing the remaining sections.

### STA4273 - Minimizing Expectations

*This seminar course introduces students to the various methodological issues at stake in the problem of optimizing expected values and leads them in a discussion of its recent developments in machine learning. The course emphasizes the interplay between reinforcement learning and Bayesian inference. While most of the readings are applied or methodological, there are topics for more theoretically-minded students. Students will be expected to present a paper, prepare code notebooks, and complete a final project on a topic of their choice.*

For more information on the syllabus, see the [course website](https://www.cs.toronto.edu/~cmaddis/courses/sta4273_w21/).
