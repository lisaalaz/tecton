import re
import traceback
from funchub.math import *
from llama3.tokens import newline_tokens, closing_bracket_tokens, opening_angular_tokens
from llama3.utils import hint_map

def tecton_generate_inference(templates, case_idx, question, funcmodel, setting, dataset, doc_dict, exemplar_dict, temperature, top_p, 
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
                          generation = cur_generation + generation
                          cur_generations.append(generation)
            
              all_generations = []
              operations = []
              for cur_generation in cur_generations:            
                for op in func_map:
                  if cur_generation.endswith(op+"("):
                    if start_length and end_length:
                        cur_generation_with_func = cur_generation                   
                    else:
                        cur_generation_with_func = cur_generation
                    funcmodel.inference_mode = "baseline"
                    prompt = templates[op].replace("[QUESTION]", question) + " " + cur_generation_with_func
                    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=0, top_p=top_p, stop_token=closing_bracket_tokens, 
                                                 disable_token=opening_angular_tokens, return_top=return_top)
                    if return_top > 0:
                        results, token_log = results
                        logs.append(token_log)     
                    generated = results[0].replace(prompt, "").replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
                    generated = re.sub("\).", ")=", generated)
                    cur_generation += generated
                    args = cur_generation.split(op)[-1].replace("=", "").replace(">", "").replace("((", "(").replace("))", ")")
                    # remove any $ in the args
                    args = args.replace("$", "")
                    # handle ^
                    args = args.replace("^", "**")
                    if ", " in args:
                        args = args.replace(", ", ";").replace(",", "").replace(";", ", ")
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
                        pass
                    else:
                      try: 
                        res = eval(f"{op[1:-1]}_{args}")

                        if len(res) < 35: # eliminate ops with too long results as they fill the context window
                          func_calls.append(f"{op}{args} = {res}")
                          start_length.append(len(cur_generation.split(op)[0]))
                          cur_generation += str(res)
                          end_length.append(len(cur_generation))
                          all_generations.append(cur_generation)
                          operations.append(op)
                      except:
                          continue
                    
              hint_list = []
              if len(all_generations) > 0:
                  hint_list = [*dict.fromkeys(["<" + x.split("<")[-1] for x in all_generations])]

              if len(hint_list) == 0:
                   cur_generation = new_generation
                   funcmodel.inference_mode = "func_embedding"
              elif len(hint_list) == 1:
                   cur_generation = all_generations[0].split("<")[0] + all_generations[0].split("=")[-1]
                   funcmodel.inference_mode = "func_embedding"
              else:
                hints = str(hint_list).replace("'", "")

                # if "gsm8k" in setting:
                #      hints_ = [*dict.fromkeys(["<" + x.split("<")[-1] for x in all_generations])] #.replace("'", "")
                #      hints = []
                #      for h in hints_: # e.g. <add>(3, 2)=5
                #         terms = h.split("(")[-1].split(")")[0]
                #         term_list = terms.split(",")
                #         term_list = [x.strip() for x in term_list]
                #         op = h.split("<")[-1].split(">")[0]
                #         result = h.split("=")[-1].strip()
                #         symbol = hint_map[op]
                #         hint = "<<" + symbol.join(term_list) + "=" + result + ">>"
                #         hints.append(hint)
                #      hints = str(hints).replace("'", "")
                     
                gen_without_hints = " ".join(all_generations[-1].split("<")[0].split(" "))
                
                #exemplar_type = "decodeallgsmformat" if "gsm8k" in setting else "decodeall"
                exemplar_type = "decodeall"
                exemplars = "\n\n".join([*dict.fromkeys([exemplar_dict[op][exemplar_type] for op in operations])])
                instructions = "\n".join([*dict.fromkeys([doc_dict[op]["overview"] for op in operations])])
                prompt = templates["choicedocs"].replace("[EXEMPLARS]", exemplars).replace(
                    "[QUESTION]", question).replace("[ANSWER]", hints + " " + gen_without_hints)

                results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p,
                                             stop_token=newline_tokens, return_top=return_top)
                       
                if return_top > 0:
                  results, token_log = results
                  logs.append(token_log)
                cur_generation = results[0].split("Answer: ")[-1].replace(hints + " ", "") 
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
            cur_generation = cur_generation.replace("\n\n", "\n")
            loop_count += 1    
            if endflag or loop_count > 15: # avoid an infinite loop
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