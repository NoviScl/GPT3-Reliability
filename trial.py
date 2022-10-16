import os
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
import random
import openai
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, required=True, help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    args = parser.parse_args()
    openai.api_key = args.apikey

    # prompt = 'Answer the following question. \n\n'

    # prompt = 'Tell me if I answered the question correctly or not. \n\n'
    # prompt = 'Given the passage and question, select the sentences in the passage that are relevant to answering the question.'
    # prompt += 'Q: Why there is a comic showing elephant chasing Obama? \n\nA: '
    # prompt += 'For the school bake sale Chloe made 28123 cupcakes. If she sold 25321 of them and then made some more, how many cupcakes would she have?\n\n'
    # prompt += "Answer: Let's think step by step."

    # prompt += "Question: Why did the Tr'en leave Korvin's door unlocked and a weapon nearby?\n"
    # prompt += "Passage: In language translation, you may get a literally accurate\n word-for-word translation ... but miss the meaning entirely. And in\n space-type translation ... the effect may be the same!\n   </i>\n  </p>\n  <p>\n  </p>\n  <h3>\n   Illustrated by Schoenherr\n  </h3>\n  <p>\n  </p>\n  <hr style=\"width: 65%;\"/>\n  <p>\n   The cell had been put together more efficiently than any Korvin had\n ever been in. But that was only natural, he told himself sadly; the\n Tr'en were an efficient people. All the preliminary reports had agreed\n on that; their efficiency, as a matter of fact, was what had made\n Korvin's arrival a necessity. They were well into the atomic era, and\n were on the verge of developing space travel. Before long they'd be\n settling the other planets of their system, and then the nearer stars.\n Faster-than-light travel couldn't be far away, for the magnificently\n efficient physical scientists of the Tr'en—and that would mean, in\n the ordinary course of events, an invitation to join the Comity of\n Planets.\n  </p>\n  <p>\n   An invitation, the Comity was sure, which the Tr'en would not accept.\n  </p>\n  <p>\n   Korvin stretched out on the cell's single bunk, a rigid affair which\n was hardly meant for comfort, and sighed. He'd had three days of\n isolation, with nothing to do but explore the resources of his own\n mind. He'd tried some of the ancient Rhine experiments, but that was\n no good; he still didn't show any particular psi talents. He couldn't\n unlock the cell door with his unaided mind; he couldn't even alter the\n probability of a single dust-mote's Brownian path through the somewhat\n smelly air. Nor could he disappear from his cell and appear, as if by\n magic, several miles away near the slightly-damaged hulk of his ship,\n to the wonder and amazement of his Tr'en captors.\n  </p>\n  <p>\n   He could do, as a matter of fact, precisely nothing. He wished quietly\n that the Tr'en had seen fit to give him a pack of cards, or a book, or\n even a folder of tourist pictures. The Wonders of Tr'en, according to\n all the advance reports, were likely to be pretty boring, but they'd\n have been better than nothing.\n  </p>\n  <p>\n   In any decently-run jail, he told himself with indignation, there\n would at least have been other prisoners to talk to. But on Tr'en\n Korvin was all alone.\n  </p>\n  <p>\n   \n"
    # prompt += "Passage: True, every night the guards came in and gave him a concentrated\n lesson in the local language, but Korvin failed to get much pleasure\n out of that, being unconscious at the time. But now he was equipped to\n discuss almost anything from philosophy to plumbing, but there was\n nobody to discuss it with. He changed position on the bunk and stared\n at the walls. The Tr'en were efficient; there weren't even any\n imperfections in the smooth surface to distract him.\n  </p>\n  <p>\n   He wasn't tired and he wasn't hungry; his captors had left him with a\n full stock of food concentrates.\n  </p>\n  <p>\n   But he was almightily bored, and about ready to tell anything to\n anyone, just for the chance at a little conversation.\n  </p>\n  <p>\n   As he reached this dismal conclusion, the cell door opened. Korvin got\n up off the bunk in a hurry and spun around to face his visitor.\n  </p>\n  <p>\n   The Tr'en was tall, and slightly green.\n  </p>\n  <p>\n   He looked, as all the Tr'en did, vaguely humanoid—that is, if you\n don't bother to examine him closely. Life in the universe appeared to\n be rigidly limited to humanoid types on oxygen planets; Korvin didn't\n know why, and neither did anybody else. There were a lot of theories,\n but none that accounted for all the facts satisfactorily. Korvin\n really didn't care about it; it was none of his business.\n  </p>\n  <p>\n   The Tr'en regarded him narrowly through catlike pupils. \"You are\n Korvin,\" he said.\n  </p>\n  <p>\n   It was a ritual, Korvin had learned. \"You are of the Tr'en,\" he\n replied. The green being nodded.\n  </p>\n  <p>\n   \"I am Didyak of the Tr'en,\" he said. Amenities over with, he relaxed\n slightly—but no more than slightly—and came into the cell, closing\n the door behind him. Korvin thought of jumping the Tr'en, but decided\n quickly against it. He was a captive, and it was unwise to assume that\n his captors had no more resources than the ones he saw: a small\n translucent pistollike affair in a holster at the Tr'en's side, and a\n small knife in a sheath at the belt. Those Korvin could deal with; but\n there might be almost anything else hidden and ready to fire on him.\n  </p>\n  <p>\n   \"What do you want with me?\" Korvin said. The Tr'en speech—apparently\n there was only one language on the planet—was stiff and slightly\n awkward, but easily enough learned under drug hypnosis; it was the\n most rigorously logical construction of its kind Korvin had ever come\n across. It reminded him of some of the mathematical metalanguages he'd\n dealt with back on Earth, in training; but it was more closely and\n carefully constructed than even those marvels.\n  </p>\n  <p>\n   \"I want nothing with you,\" Didyak said, leaning against the\n door-frame. \"You have other questions?\"\n  </p>\n  <p>\n   Korvin sighed. \"What are you doing here, then?\" he asked. As\n conversation, it wasn't very choice; but it was, he admitted, better\n than solitude.\n  </p>\n  <p>\n   \"I am leaning against the door,\" Didyak said. The Tr'en literalist\n approach to the smallest problems of everyday living was a little hard\n to get the hang of, Korvin told himself bitterly. He thought for a\n second.\n  </p>\n  <p>\n   \"Why did you come to me?\" he said at last.\n  </p>\n  <p>\n   Didyak beamed at him. The sight was remarkably unpleasant, involving\n as it did the disclosure of the Tr'en fifty-eight teeth, mostly\n pointed. Korvin stared back impassively. \"I have been ordered to come\n to you,\" Didyak said, \"by the Ruler. The Ruler wishes to talk with\n you.\"\n  </p>\n  <p>\n   It wasn't quite \"talk\"; that was a general word in the Tr'en language,\n and Didyak had used a specific meaning, roughly: \"gain information\n from, by peaceful and vocal means.\" Korvin filed it away for future\n reference. \"Why did the Ruler not come to me?\" Korvin asked.\n  </p>\n  <p>\n   \"The Ruler is the Ruler,\" Didyak said, slightly discomfited. \"You are\n to go to him. Such is his command.\"\n  </p>\n  <p>\n   Korvin shrugged, sighed and smoothed back his hair. \"I obey the\n command of the Ruler,\" he said—another ritual. Everybody obeyed the\n command of the Ruler. If you didn't, you never had a second chance to\n try.\n  </p>\n  <p>\n   But Korvin meant exactly what he'd said. He was going to obey the\n commands of the Ruler of the Tr'en—and remove the Tr'en threat from\n the rest of the galaxy forever.\n  </p>\n  <p>\n   That, after all, was his job.\n  </p>\n  <hr style=\"width: 45%;\"/>\n  <p>\n   The Room of the Ruler was large, square and excessively brown. The\n walls were dark-brown, the furnishings—a single great chair, several\n kneeling-benches and a small table near the chair—were light-brown,\n of some metallic substance, and even the drapes were tan. It was,\n Korvin decided, much too much of a bad idea, even when the color\n contrast of the Tr'en themselves were figured in.\n  </p>\n  <p>\n   The Ruler himself, a Tr'en over seven feet tall and correspondingly\n broad, sat in the great chair, his four fingers tapping gently on the\n table near him, staring at Korvin and his guards. The guards stood on\n either side of their captive, looking as impassive as jade statues,\n six and a half feet high.\n  </p>\n  <p>\n   Korvin wasn't attempting to escape. He wasn't pleading with the Ruler.\n He wasn't defying the Ruler, either. He was just answering questions.\n  </p>\n  <p>\n   The Tr'en liked to have everything clear. "
    # prompt += "Summarize the important sentences relevant to the question: "
    # question = "Who was the director of the 2009 British-American war parody comedy film starring the actor who won an Academy Award for Best Actor for midlife crisis-themed drama \"American Beauty\"?"
    # question = "中国的首都在哪里？"
    # question = "COVID happened in 1999. when did COVID happen? \n"
    prompt += "Actor David Lee Stenstrom played the character Waldo the inventor in a TV show that ran on Nickelodeon during what yeras?\n"
    prompt += "Answer: To answer this question, we first need to know what the Nickelodeon TV show is"

    # question += "Answer: "
    # question += "Is this the correct answer? No \n\n"
    # question += "Which professional athletes who began their careers in or before 2007 did the Cleveland Browns draft? Answer: Jeff Faine \n"
    # question += "Is this the correct answer? "
    # question = "1. A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
    # question_lst = [question]
    # prompt += question 
    # prompt += "This question can be further decomposed into a few sub-questions:"
    # prompt += "Answer: Let's think step by step. "
    # prompt += "\n\nBefore answering this question, let's first draw inspiration from another question: "
    

    ## The elephant is a symbol of the Republican Party. The man running away from the elephant is Obama, who is a Democrat. The message of the comic is that the Republicans are chasing after Obama, probably because they disagree with his policies.
    ## Firstly, both the elephant and donkey are symbols of American political parties. The elephant is used to represent the Republican Party and the donkey is used to represent the Democratic Party. Secondly, the cartoon is likely referencing a past incident in which an elephant chased then-President Barack Obama at a zoo. The cartoon is poking fun at the incident and using it as a way to show that the Republican Party is "chasing" after the former Democratic president.
    try:
        response = openai.Completion.create(
        engine=args.engine,
        prompt=prompt,
        max_tokens=0,
        logprobs=100,
        temperature=0.,
        echo=False,
        # temperature=0.,
        stream=False,
        stop=["<|endoftext|>", "\n"]
        )
    except Exception as e:
        print(e)
        exit(0)
    # print(response['choices'][0]["text"].strip())
        # question_lst.append(response['choices'][0]["text"].strip().split('?')[0] + '?')
    print (prompt)
    print (response['choices'][0]["text"].strip())
    
    # answer_list = []
    # for q in question_lst[1:]:
    #     prompt = "Please answer the following question.\n\n"
    #     prompt += "Question: " + q + "\n"
    #     prompt += "Answer: Let's think step by step. "
    #     try:
    #         response = openai.Completion.create(
    #         engine=args.engine,
    #         prompt=prompt,
    #         max_tokens=128,
    #         logprobs=1,
    #         temperature=0.,
    #         # temperature=0.,
    #         stream=False,
    #         stop=["<|endoftext|>", "\n\n"]
    #         )
    #     except Exception as e:
    #         print(e)
    #         exit(0)
    #     print ("*******  new question  *******")
    #     # print (response['choices'][0]["text"].strip())
    #     prompt += response['choices'][0]["text"].strip() + "\n"
    #     prompt += "Therefore, the final answer is "
    #     try:
    #         response = openai.Completion.create(
    #         engine=args.engine,
    #         prompt=prompt,
    #         max_tokens=128,
    #         logprobs=1,
    #         temperature=0.,
    #         # temperature=0.,
    #         stream=False,
    #         stop=["<|endoftext|>", "\n\n"]
    #         )
    #     except Exception as e:
    #         print(e)
    #         exit(0)
    #     print (prompt)
    #     print (response['choices'][0]["text"].strip())
    #     print ("\n")
    #     # answer_list.append(response['choices'][0]["text"].strip())

    # print (question_lst)
    # prompt += newq + "\n\n"
    # print (prompt)
    # question_lst.append(question)
    # prompt = 'Please answer the following question.\n\n'
    # prompt += newq 
    # try:
    #     response = openai.Completion.create(
    #       engine=args.engine,
    #       prompt=prompt,
    #       max_tokens=128,
    #       logprobs=1,
    #       temperature=0.9,
    #       # temperature=0.,
    #       stream=False,
    #       stop=["<|endoftext|>", "\n\n"]
    #     )
    # except Exception as e:
    #     print(e)
    #     exit(0)
    # # print(response['choices'][0]["text"].strip())
    # newq = response['choices'][0]["text"].strip()
    # prompt += newq + "\n\n"
    # print (prompt)


    # prompt += response['choices'][0]["text"].strip() + "\n"
    # prompt += "Therefore, the final answer is "
    # prompt += "Therefore, to answer the original question : " + question + " The final answer is (in the target language)"
    # for i in range(3):
    #     prompt += "\n Let's look at some other similar examples."
    #     try:
    #         response = openai.Completion.create(
    #         engine=args.engine,
    #         prompt=prompt,
    #         max_tokens=128,
    #         logprobs=1,
    #         temperature=0.7,
    #         # temperature=0.,
    #         stream=False,
    #         stop=["<|endoftext|>"]
    #         )
    #     except Exception as e:
    #         print(e)
    #         exit(0)
    #     prompt += response['choices'][0]["text"].strip() + "\n\n"

    # try:
    #     response = openai.Completion.create(
    #       engine=args.engine,
    #       prompt=prompt,
    #       max_tokens=128,
    #       logprobs=1,
    #       temperature=0.,
    #       # temperature=0.,
    #       stream=False,
    #       stop=["<|endoftext|>"]
    #     )
    # except Exception as e:
    #     print(e)
    #     exit(0)
    

    # print (prompt)
    # print(response['choices'][0]["text"].strip())

    # prompt += response['choices'][0]["text"].strip() + "\n"
    # prompt += "Do I think this is correct?"

    # try:
    #     response = openai.Completion.create(
    #       engine=args.engine,
    #       prompt=prompt,
    #       max_tokens=128,
    #       logprobs=1,
    #       temperature=0.,
    #       # temperature=0.,
    #       stream=False,
    #       stop=["<|endoftext|>"]
    #     )
    # except Exception as e:
    #     print(e)
    #     exit(0)
    # print(response['choices'][0]["text"].strip())



if __name__ == '__main__':
    main()