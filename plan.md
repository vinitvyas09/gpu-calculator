

I'm improving my GPU calculator spec by researching existing tools.                                                                          
                                                                                                                                               
  The spec: spec/gpu-calculator/llm-training-gpu-calculator-spec.md                                                                            
  The source list: spec/gpu-calculator/research-sources.md                                                                                     
                                                                                                                                               
  For each source in research-sources.md (in order, top to bottom):                                                                     
  1. Use @sw-researcher to deep-dive the source                                                                                                
  2. Pass findings to @sw-architect to evaluate, apply changes, and git commit                                                                 
  3. Move to the next unchecked source immediately                                                                                             
                                                                                                                                               
  Do NOT stop between sources. Run through ALL Tier 1 sources autonomously.                                                                    
  Skip it if a source is unreachable or you hit an ambiguity that genuinely                                                                  
  blocks the entire pipeline.                                                                                                                  
                                                                                                                                               
  Start now with the first unchecked source. Keep going until all are done in 'Mixed Precision' and 'Scaling Laws' and 'Inference (for reference)' ONLY


remember multiple agents are working in parallel in other terminals. so for the sw architect, it needs to rely ONLY on the info/context provided by the research agent. If it is looking at the files, make sure it ONLY looks at the section it is responsible for and then do git commit accordingly. ONLY COMMIT THE THING YOU APPROVED ADN WAS WORKED on BY your agent. SOunds good? THIS IS A VERY CRITICAL POINT.