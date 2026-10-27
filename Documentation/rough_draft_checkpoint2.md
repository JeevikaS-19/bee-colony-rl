# checkpoint 2: 

the previous paper covered the design, biological grounding and what had been built and tested in isolation. 
### since then: 
working on the same architecture and problems, the new issues i hit 

## what the design settled into: 
1. sensing the potential, in the first version, i wrote that the read takes the max over nearby cells, what was built: 
    #pheromone_table.py, line 110
    sensed[trace_type] = max(sensed[trace_type], effective_intensity)

    the sensing function G is: 
        - location --> ego position(x,y)
        - nearby cells --> all grid cells within sensor_radius = 10m
        - Spatial weight: Gaussian falloff with σ = 6.0 m, exp(-(dist²) / (2σ²))
        - Temporal decay: exp(-decay_rate * Δt), where Δt is training steps since write
        - Final reading: max over cells of (vigor × spatial_weight × temporal_decay)

    Why β changed: a pilot run showed that with only novice bees (who write warnings but not successes), the shaping was about 0.014 per step when raw reward was ~0.1 per step—a 700:1 ratio. The design called for shaping to be about 10% of raw reward. So you measured β on the 60k checkpoint using the measure_beta.py script: ran 10 stochastic rollouts at β=1, found that β=2.86 brings mean|F| to ~10% of mean|r|. This β is frozen in the spec.
2. reputation -> changed then settled 
    initially, hard caste thresholds at E_t = 0.3 and 0.8, fixed vigor values 

    E_t is not a simple counter, it's an expotentially weighted moving average of arrival rate

    Why: i replayed the outcomes of one 100k run (10 arrivals, 164 crashes, 26 timeouts in 200 episodes) through a simple step-based rule, and it never promoted anyone. With EWMA, smoothing 0.1 promotes at episode 181 and reaches FORAGER by the end. The step sizes 0.05 up / 0.10 down never worked because with a 5% arrival rate, the rule is designed for drift, not jumps. EWMA makes E a proxy for the rolling arrival rate: E ≈ 0.3 means "about 30% arrivals lately."

    
