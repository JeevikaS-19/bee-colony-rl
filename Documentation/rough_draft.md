# Rough draft - Article 1 

## Starting point

The first version of this project i worked on was with sumo simulator, a 2d simulator that allowed me to take lidar and the cams as input, ideal for testing out self driving algorithms. Spent a good portion of the year learning about the basics of Reinforcement learning from *stanford lectores* and *mit lectures* where in the course they taught us how to make a base model. 

After that was done, I worked towards building it on a 3d simulator, meta drive. Where i worked on implementing SAC algorithm. 

### The challenges i faced, which is what pushed me to this idea: 
1. While training, processing the data was messy, working on trying ot process 1 million iterations was pushing my system to the max, took me 2 days to train it. 
    - There had to be an easier way right? now sure that was a hardware issue, the fact that my system lacked a dedicated GPU might have contibuted to the issue. But what about others, not everyone can work with established companies to process such chunky data. 
    - Compressing the data would be risking the quality, i needed the car to atleast see the lanes and what's around it. 
    - What if we had a multi tread system, kind of, multiple agents processing the data and sharing the memory, rather than one model struggling to retain what it learned. What if we had others helping eachother to understand the roads as well. 
    - I'll address the above point shortly. 
2. The exploaration would floor after a new runs, which was fixable, now even after we fixed the alpha parameter, the *model* was struggling to complete the course. Observation, it would drive just fine, cover about 30% of the episode. But after that, it would jitter, almost like it suddendly forgot what to do. 
    - i never quite understood what caused it, needed more training? or while training it got stuck at that point, it never understood that the reward to finish the episode was huge, becuase it never got it. 
    - no amount of fine tuning helped it to cross past this point, it wasnt training properly. After spending two weeks on trying to understand why, I was getting nowhere with it. I had to pause and take a step back. 
    - (there was another reason, go through prior articles)
3. what were the other issue? i don't remember 

### Now, talking about how I come to this architecture, the *bee colony*. 

I had read about the advancements in the chinese infrastructure when it came to self driving cars, an idea that was yet to be implemented. Now the biggest issue with autonomous cars, be it tesla or waymo, the currently leading automonous cars in the market, they relied on rules, both would struggle to see oncomming diversions if not in maps. China wanted to improve on that by building a smart infrstructure around the car. Diversion up ahead, the car gets the signal to override the maps. 

It wasn't just the infrastructure, they wanted cars to communicate with eachother, *crach ahead*, *warnings* passed to the cars coming that way. Now implementing multi-agent communication is a huge burden on the overhead servers. It wasn't the idea that inspired me, more like the logic. I wasn't solving the same problem. 

But what if I could implement training that way, while training it's multiple agents communicating and guiding/helping other cars that are stuck. 

That raises a question, I just mentioned that multi-agent communication is a burden, that led me to think about how would I not make it a burden. How do I make it lightweight? 

*Bees*, *bee colonies*, we've all read about how thousands of bees work togther in the hive. The work that goes into foraging honey or a new home. I had so many questions. I didn't think about this in terms of the project initially, but the more I looked into it, what feed my curiosity might have just solved my problem. 

### Getting more into the *bee colony* architecture: 
1. **The Biological Foundation**: 
    1. Staring with how do bees communicate, thousands of bees, how do they pass on the information of where the honey is? **Waggle Dance**
        - Inside the hive, the returning bees share geographic coordinates using symbolic, figure-eight movement - Waggle dance
        - Operates as an abstract  communication protocol that translates 3D spacial vectors into tactical/vibrational signals. 
        *This what caught my attention, multi-agents need to pass on chunky data, how would I simplify it?*
        - **Direction Encoding**: on the vertical comb face the bee transposes celestial navigation into a gravitational reference frame. The angle of the waggle run corresponds to the target's angle relative to the sun's azimuth outside. 
        - **Distance Encoding**: The duration of the waggle run represents the flight distance to the goal. 
        - **Quality and vigor encoding**: the total number of dance circuits, the tempo of the return loops and enthusiam of the dance communicate the profitability or desirability of the location. 
        *The first thing that came to my mind here, was the reward system*
        -(add the sources that elaborate more on the biological components)
        - The AV fleet analogy: Rather than sending unstructed high-bandwidth sensor streams across the vehicle to everything network, individual vehical agents compress local anamoly detections into a structured message, such as, direction *heading relative to the road geometry*, distance *location along the corridor* and confidence/vigor *anomaly severity score*
    
    2. Taking inspiration from how SAC algorithm works and how pheromons work. In the field of robotics, they use the concept of pheromons. Other models leave a trail, like a warning signal. What if i could do that with the training models, it encounters a turning, it needs to learn that it has to turn at that point. 
        - warning pheromons would trap me again, so instead of the agents dropping a warning telling the others that this stretch can result in a penalty, it should act as a guide. 
        - What ants do it that the pheromons guide the ants. In this case, an agent that attempted tp turn, knows that it oversteered or understeered to the right, so what if the next one learns that if needs to take a sharper turn or wider turn. 
        - setting a warning pheromon would result in the model not attempting that stretch, it would look at just the penalty and freeze. *a problem i faced in the previous model*
        - (i'll elaborate more)
        - pheromon table and it's working 

    3. Another issue came up, one i had faced in the previous version. How do I know which bee to listen to, what if the whole communication (shared memory) is flooded with the bad? The pheromon table wouldn't now what to prioritise. 
        - Age Polyethism and decentralised hierarchy (come back to this later)
        - younger ones learn for bees that have been out, only bees that have experience will commit to the pheromon table. (i'll explain later)
        - **AV Fleet Analogy:** Vehicle nodes in the fleet do not possess fixed, static influence. Instead, individual vehicles develop **dynamic authority weights** based on their operational history and prediction accuracy (mimicking age polyethism and octopamine modulation). A newly initialized vehicle node acts as an uncalibrated "novice," while vehicles with long, error-free operational histories act as high-vigor "scouts" whose anomaly signals carry higher weight.

    4. When it comes to the reward system, how do we get the models to want to explore or move ahead? That's where the vigor comes in play
        - Collective decision-making in a hive relies on a delicate balance between **positive feedback (excitation/recruitment)** and **negative feedback (inhibition/stop signals)**
        - **AV Fleet Analogy:** Positive success traces ("turning guide" pheromones) reinforce safe, high-efficiency routes through complex intersections, while negative stop signals ("warning" pheromones) act as localized hazard flags dropped upon detecting a hard brake, curb strike, or obstacle. By allowing positive and negative signals to interact through decay and spatial filtering, the fleet achieves **quorum consensus** on localized hazards without deadlocking or suffering from false-alarm cascades.

### a few things i had to test and make sure was working
1. Parallel infrastructure and subprocess management 
    - before i worked on the project, i needed to check if metadrive even allowed it, so in the file *test_parallel_pluming.py* (we do it) 
    - implementing a non-blocking multiprocessing.Queue architecture with active parent draining and sentinel poison pills "DONE" to prevent queue buffer congestion deadlocks 

2. Before deploying multiple bees, i needed to make sure one bee could drive properly, that one bee that guides the rest. 
    - *run_sac_sandbox.py*, we isloate a single sandbox bee to confirm the policy forward passes and tensor boundaries
    - taking the previous model we built, with the same stochastic SAC Actor  algorithm, (do i need to elaborate on this?)
    - Twin critics(SACCcritic), dual Q-networks taking state-action pairs(s,a) to combat value overestimation bias. 
    - replay buffer, stored transistion tuples for offline memory consolidation, breaking frame correlation during gradient updates 


### Pheromone table
- *pheromone_table.py* and *test_pheromone_table.py*
- Designed and verified the spatial blackboard representing the hive's collective memory floor
    - 2-meter spatial bucketing: Discretized continuous 2d coordinates(x,y) into integer grid cells(_to_grid_key) matching lane-width resolution
    - Decoupled Parallel channels: the inital structure i was worried about what the model would see, what if it was flooded with warning and the success (turning guide) was buried? To overcome that, I seperated the "success" and the "warning" channels so safe corridors and danger flags coexist without overwriting each other
    - As i disvussed earlier, I wanted to give more importance to experianced bees or to ones with more vigor. In a hive, for example, bee A tells the hive about place A and there's another bee, bee B, that is found place B. How would they decide which one? they have a dance off, where they head bump eachother and the one with the most vigor is the one that guides the rest. Similarly, I implemented Max-vigor Overwrite Rule. 
    - Now in real life, pheromons dessolve after a certain period, to do that, we use spatial and temporal decay. Applied Gaussian distance falloff for localized sensing footprints and exponential time evaporation. 
    - verified the grid bucketing, overwrite gating, double-channel isolation, sensor falloff ranges and clean air reads
    


    
