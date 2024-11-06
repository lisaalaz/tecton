import itertools
import traceback
import re
from funchub.math import *
from llama3.tokens import newline_tokens, option_tokens, num_tokens, closing_bracket_tokens, opening_angular_tokens
from llama3.utils import bias_gsm8k, bias_funcqa, uppercase_alphabet

def tecton_score_inference(templates, case_idx, question, funcmodel, setting, dataset, doc_dict, exemplar_dict, temperature, top_p, 
                                         max_gen_len, return_top=5): 

    cur_generation = ""
    cur_generation_with_func = ""
    start_length = []
    end_length = []
    logs = []
    funcmodel.inference_mode = "func_embedding"
    func_map = list(funcmodel.func_dict.keys())
    endflag = False
    
    try:
        results = [] 
        func_calls = []
        loop_count = 1
        while True: # loop until break
            prompt = templates["generalnew"].replace("[QUESTION]", question) + cur_generation
            results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=newline_tokens,
                                         return_top=return_top)
            if return_top > 0:
                results, token_log = results 
                logs.append(token_log)

            new_generation = results[0].replace(templates["generalnew"].replace("[QUESTION]", question), "").replace("<|begin_of_text|>", "")
            found_toolkens = any([x[1][i][0] >= 128256 for x in token_log for i in range(len(x[1]))])

            if found_toolkens and "####" not in new_generation: 
              cur_generations = []
              for i in range(len(token_log)):
                  for t in token_log[i][1]:
                      if (t[0] >= 128256):
                          toolken = t[0]
                          token_list = [x[0] for x in token_log[:i]] + [toolken]
                          generation = funcmodel.decode_list(token_list)                          
                          cur_generations.append(generation)       
              
              all_generations = []
              operations = []
              for cur_gen in cur_generations:
            
                for op in func_map:
                  if cur_gen.endswith(op+"("):
                    if start_length and end_length:
                        cur_generation_with_func = cur_gen
                    else:
                        cur_generation_with_func = cur_gen
                    funcmodel.inference_mode = "baseline" 
                    complete_answer = cur_generation + cur_generation_with_func
                    prompt = templates[op].replace("[QUESTION]", question) + " " + complete_answer.replace("\n", " ")
                    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=0, top_p=top_p, stop_token=closing_bracket_tokens, 
                                                 disable_token=opening_angular_tokens, return_top=return_top) 
                    if return_top > 0:
                        results, token_log = results
                        logs.append(token_log)
                    generated = results[0].replace(prompt, "").replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
                    generated = re.sub("\).", ")=", generated)
                    cur_gen += generated
                    args = cur_gen.split(op)[-1].replace("=", "").replace(">", "").replace("((", "(").replace("))", ")")
                    # remove any $ in the args
                    args = args.replace("$", "")
                    # handle ^
                    args = args.replace("^", "**")
                    if ", " in args:
                        args = args.replace(", ", ";").replace(",", "").replace(";", ", ") # this leaves ", " unchanged but eliminates commas without spaces
                    args = args.replace(" ", "")
                    if "(" not in args or ")" not in args:
                        raise Exception("invalid args")
                    # handle %
                    if '%' in args:
                        temp = args.split("(")[1].split(")")[0].split(",")
                        for arg_i, arg in enumerate(temp):
                            if "%" in arg:
                                arg = arg.replace("%", "").strip()
                                arg = str(float(arg) / 100)
                            temp[arg_i] = arg
                        args = f"({', '.join(temp)})"                   
                    if (
                         op not in ["<log>", "<ln>", "<sqrt>"] and "," not in args
                        ) or (
                            op in ["<choose>", "<permutate>", "<remainder>", "<lcm>", "<gcd>"] and "." in args
                            ):
                        continue
                    else:
                      try: 
                        res = eval(f"{op[1:-1]}_{args}") 
                        if len(res) < 35: # eliminate ops with too long results as they fill the context window
                          func_calls.append(f"{op}{args} = {res}") 
                          start_length.append(len(cur_generation.split(op)[0]))
                          cur_gen += str(res) 
                          end_length.append(len(cur_gen))
                          if ")=" not in cur_gen and ") =" not in cur_gen:
                             cur_gen = cur_gen.replace(")", ")=")
                          all_generations.append(cur_gen)
                          operations.append(op)
                      except:
                        continue

              if len(all_generations) == 0:
                     cur_generation += new_generation
                     funcmodel.inference_mode = "func_embedding"
              else:
                all_generations = [all_generations[0]] + [x for x in all_generations[1:] if ("<" and ")=") in x or "<" not in x]
                tool_ops = ["<" + x.split("<")[-1] for x in all_generations] #[*dict.fromkeys()] #.replace("'", "")
                unique_tools = [*dict.fromkeys(tool_ops)]
                unique_tools = unique_tools[:4]
                      
                try:
                     unique_tool_continuations = dict(zip(["<" + x.split("<")[1].split(")")[0] + ")" for x in all_generations], all_generations))
                except:
                   continue

                if len(unique_tool_continuations) > 4:
                   unique_tool_continuations = dict(itertools.islice(unique_tool_continuations.items(), 4))

                toolchoice_options = dict(zip(uppercase_alphabet[:len(unique_tools)], unique_tools))
                toolcompletionchoice_options = dict(zip(uppercase_alphabet[:len(unique_tool_continuations.values())], unique_tool_continuations.values()))
                options = toolcompletionchoice_options

                if len(options) == 1:
                   cur_generation += toolcompletionchoice_options["A"].split("<")[0] + toolcompletionchoice_options["A"].split("=")[-1].replace(">>", "")

                else:
                      template_setting = "toolcompletionchoice"
                      prompt = templates[f"{template_setting}{len(options)}"
                                         ].replace("[QUESTION]", question).replace("[ANSWER]", cur_generation.replace("\n", "")).replace("[OPTIONS]", "\n".join(
                                          [f"{k}: {v}" for k,v in options.items()]))
                      
                      probs = funcmodel.score([prompt], enable_only_token=option_tokens[:len(options)])
                      if setting == "gsm8k":
                         biases = bias_gsm8k[f"{len(options)}_options"]
                      elif setting == "funcqa":
                         biases = bias_funcqa[f"{len(options)}_options"]
                      equal_split_prob = 1.0 / len(options)
                      for letter in probs:
                         probs[letter] += equal_split_prob - biases[letter]
                      choice = max(probs, key=probs.get)

                      selected_continuation = options[choice]
                      tool_to_delete = toolchoice_options[choice].split("=")[0] + "="
                      selected_continuation = selected_continuation.replace(tool_to_delete, "").replace(">>", "")
                      cur_generation += " " + selected_continuation

                prompt = templates["generalnew"].replace("[QUESTION]", question) + cur_generation
                results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, stop_token=newline_tokens, 
                                             return_top=return_top, disable_token=num_tokens)
                if return_top > 0:
                     results, token_log = results
                     logs.append(token_log)
                cur_generation = results[0].replace(templates["generalnew"].replace("[QUESTION]", question), "").replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
                  
              funcmodel.inference_mode = "func_embedding"
            
            else:
                if "####" in new_generation:
                  if "<" in new_generation.split("####")[-1]:
                     funcmodel.inference_mode = "baseline"
                  else:
                     cur_generation = new_generation
                     endflag=True
                else:
                   cur_generation = new_generation
            loop_count += 1    
            if endflag or loop_count > 15: # to avoid getting stuck in infinite loop
                break
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": "success"
        }

    except Exception:
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(traceback.format_exc())
        }
    return log