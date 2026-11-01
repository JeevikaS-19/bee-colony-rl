MULTI_BEE_TRAINING():
  # Shared state
  shared_agent = SACAgent(obs_dim=259, action_dim=2)
  shared_buffer = ReplayBuffer(capacity=200000)
  pheromone_table = PheromoneTableV2()
  
  # Worker function (runs in each subprocess)
  WORKER(worker_id, shared_agent, shared_buffer, pheromone_table):
    env = create_env()
    local_agent = copy(shared_agent)  # read current weights each episode
    
    for episode in range(num_episodes):
      obs = env.reset()
      done = False
      step = 0
      
      while not done:
        action = local_agent.sample_action(obs)
        obs_next, reward, done, info = env.step(action)
        
        # Read pheromone (shaping)
        if reading_enabled:
          readout = dance_reader.read_corridor(obs['position'], obs['heading'], step)
          phi_curr = shaping_calculator.compute_potential(obs['heading'], obs['speed'], readout)
          phi_next = shaping_calculator.compute_potential(obs_next['heading'], obs_next['speed'], readout)
          shaped_reward = reward + (gamma * phi_next - phi_curr)
        else:
          shaped_reward = reward
        
        shared_buffer.push((obs, action, shaped_reward, done))
        obs = obs_next
        step += 1
      
      # Write pheromone at episode end
      if done and info['arrive_dest']:
        write_success_path(pheromone_table, path, worker_id, step)
      elif done and info['crash']:
        write_crash_point(pheromone_table, crash_pos, crash_type, worker_id, step)
      
      # Sync weights from shared_agent
      local_agent.load_state(shared_agent.state_dict())
  
  # Launch N workers in parallel
  workers = [spawn(WORKER, args=(i, shared_agent, shared_buffer, pheromone_table)) for i in range(num_bees)]
  
  # Main learner loop
  for step in range(total_steps):
    if shared_buffer.size > batch_size:
      batch = shared_buffer.sample(batch_size)
      shared_agent.update_parameters(batch)
    
    # Periodically sync workers (they pull new weights)
    if step % 1000 == 0:
      broadcast(shared_agent.state_dict() to all workers)
  
  # Cleanup
  terminate_all_workers()