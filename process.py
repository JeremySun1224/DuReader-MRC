import json


if __name__ == "__main__":
    # predictions = {}
    # with open("predictions_0518.json", "w", encoding="utf-8") as fs:
    #     with open("nbest_predictions_0512.json", "r", encoding="utf-8") as fw:
    #         data = json.loads(fw.read())
    #         for k, v in data.items():
    #             idx = 0
    #             ans = v[idx]["text"]
    #             while len(ans) > max_len:
    #                 try:
    #                     ans = v[idx + 1]["text"]
    #                     idx += 1
    #                 except:
    #                     print(v, idx)
    #                     ans = v[idx]["text"]
    #                     break
    #             predictions[k] = ans
    #     fs.write(json.dumps(predictions, indent=4, ensure_ascii=False) + "\n")

    predictions = {}
    with open("dev_predictions.json", "w", encoding="utf-8") as fs:
        with open(
            "roberta_large_brightmart_output/DuReader/nbest_predictions_.json",
            "r",
            encoding="utf-8",
        ) as f1:
            with open(
                "roberta_large_adv_output/DuReader/nbest_predictions_.json",
                "r",
                encoding="utf-8",
            ) as f2:
                with open(
                    "roberta_large_brightmart_output2/DuReader/nbest_predictions_.json",
                    "r",
                    encoding="utf-8",
                ) as f3:
                    with open(
                        "roberta_large_adv_output2/DuReader/nbest_predictions_.json",
                        "r",
                        encoding="utf-8",
                    ) as f4:
                        data1 = json.loads(f1.read())
                        data2 = json.loads(f2.read())
                        data3 = json.loads(f3.read())
                        data4 = json.loads(f4.read())
                        for idx, v_lst1 in data1.items():
                            v_lst2 = data2[idx]
                            v_lst3 = data3[idx]
                            v_lst4 = data4[idx]
                            # l1, l2, l3 = len(v_lst1), len(v_lst2), len(v_lst3)
                            v_lst = v_lst1 + v_lst2 + v_lst3 + v_lst4
                            mem = {}
                            for i, item in enumerate(v_lst):
                                # if i < l1:
                                #     weight = 0.4
                                # elif i < l2:
                                #     weight = 0.4
                                # else:
                                #     weight = 0.2
                                if item["text"] in mem:
                                    mem[item["text"]] += item["probability"]
                                else:
                                    mem[item["text"]] = item["probability"]
                            predictions[idx] = max(mem, key=mem.get)
                        fs.write(
                            json.dumps(predictions, indent=4, ensure_ascii=False) + "\n"
                        )
