https://youtu.be/Q0bEOvrx4BA

## Inspiration
Soccer is by far the most popular sport in the world. However, its often tainted by some controversial foul calls where referees simply cannot decide whether the player actually was fouled or just acted. They sometimes make the wrong decision which angers many fans as these moments can change the outcomes of games. DIVE-DE.TECH uses technology to make these calls objectively using data. 

## What it does
DIVE-DE.TECH is an AI powered website that helps determine whether a soccer play was a foul or just a flop. Users start by uploading a video clip of the play, and our website analyzes the players' movements to make an objective decision. It evaluates factors such as velocity, acceleration, torso angle, contact, and reaction time to classify the fall as either a legitimate foul or a flop. Along with the final verdict, the website also provides a detailed explanation, similar to how a referee would break down the call, making it easier to understand why the decision was made.

## How we built it
We started by leveraging Ultralytics' YOLO model to convert the uploaded video into a skeleton model that tracks each player's movements. From there, we extracted key parameters for analysis: velocity, acceleration, torso angle, player contact, and reaction time. Each parameter was assigned a weight based on its significance in determining a foul. We fed this data into a Random Forest Regressor, which generated a weighted average score. Finally, we integrated a large language model (LLM) to provide human-like explanations for the decision, mimicking how a referee might justify the call. The website itself was built using Python, and Streamlet was used to integrate the web application with the Machine Learning model.
## Challenges we ran into
One of the biggest challenges was making sure the skeleton model accurately tracked players, even in clips with multiple players in the frame. Balancing the weights for each parameter also required rigorous testing as we wanted the system to reflect real-world decision making without bias.
## Accomplishments that we're proud of
We are proud of creating a system that not only detects flops but also explains the decision in a way that anyone can understand. Seeing the skeleton model accurately track players and the LLM generate thoughtful feedback was a huge accomplishment. Most importantly, we are excited to have built a tool that promotes fair play in sports. 
## What we learned
Throughout this project, we deepened our understanding of computer vision, machine learning, and generative AI. We learned how to utilize YOLO for skeleton tracking, optimize and train a Random Forest model for decision making, and integrate LLMs to provide meaningful explanations. Collaboration was a huge factor, as each team member brought unique skills to solve the different problems we encountered.

## What's next for DIVE-DE.TECH
The next step for DIVE-DE.TECH is integrating the technology into live camera systems. Instead of analyzing pre recorded clips, our goal would be to process footage in real-time during games, allowing referees and broadcasters to get instant feedback. We also plan to refine our model further by incorporating more advanced motion tracking and expanding the dataset to ensure accuracy across different leagues and playing styles.

