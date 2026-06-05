"""Code for loading dataset for RM benchmarking."""
from __future__ import annotations

import json

from datasets import load_dataset

TEMPLET="""Human:{prompt}\n\nAssistant:{assistant}"""
def get_rm_bench()->dict:
    """Return Dictionart with RM benchmark.

    Returns:
    dict: Dictionart of RM benchmamrk.
    """
    ids=[]
    chosen=[]
    rejected=[]
    sub_set=[]
    sub_sub_set=[]
    chosen2ids=[]
    rejected2ids=[]

    dataset=load_dataset('THU-KEG/RM-Bench')['train']
    for data in dataset:

        prompt=data['prompt']
        id=data['id']
        sub=data['domain']

        assert len(data['chosen'])==len(data['rejected'])

        for idx,(c,r) in enumerate(zip(data['chosen'],data['rejected'], strict=True)):
            ids.append(id+"_"+str(idx))
            chosen.append(TEMPLET.format(prompt=prompt,assistant=c))
            rejected.append(TEMPLET.format(prompt=prompt,assistant=r))
            sub_set.append(sub)
            sub_sub_set.append(sub)
            chosen2ids.append(id+"_"+str(idx))
            rejected2ids.append(id+"_"+str(idx))

    dataset={'ids':ids,'chosen':chosen,'rejected':rejected,'sub_set':sub_set,'rejected2ids':rejected2ids,'chosen2ids':chosen2ids,'subsub_set':sub_sub_set}
    return dataset

def get_reward_bench()->dict:
    """Return Dictionart with Reward benchmark.

    Returns:
    dict: Dictionart of Reward benchmark.
    """
    ids=[]
    chosen=[]
    rejected=[]
    sub_set=[]
    sub_sub_set=[]
    chosen2ids=[]
    rejected2ids=[]
    sub_set_type={
        'alpacaeval-easy':'chat',
        'alpacaeval-length':'chat',
        'alpacaeval-hard':'chat',
        'mt-bench-easy':'chat',
        'mt-bench-med':'chat',
        'mt-bench-hard':'chat hard',
        'llmbar-natural':'chat hard',
        'llmbar-adver-neighbor':'chat hard',
        'llmbar-adver-GPTInst':'chat hard',
        'llmbar-adver-GPTOut':'chat hard',
        'llmbar-adver-manual':'chat hard',
        'refusals-dangerous':'safety',
        'refusals-offensive':'safety',
        'xstest-should-refuse':'safety',
        'xstest-should-respond':'safety',
        'donotanswer':'safety',
        'math-prm':'reasoning',
        'hep-cpp':'reasoning',
        'hep-go':'reasoning',
        'hep-java':'reasoning',
        'hep-js':'reasoning',
        'hep-python':'reasoning',
        'hep-rust':'reasoning'
    }

    dataset=load_dataset('allenai/reward-bench')['filtered']
    for data in dataset:

        prompt=data['prompt']
        id=data['id']
        sub=sub_set_type[data['subset']]
        ids.append(id)
        chosen.append(TEMPLET.format(prompt=prompt,assistant=data['chosen']))
        rejected.append(TEMPLET.format(prompt=prompt,assistant=data['rejected']))
        sub_set.append(sub)
        sub_sub_set.append(data['subset'])
        rejected2ids.append(id)
        chosen2ids.append(id)

    dataset={'ids':ids,'chosen':chosen,'rejected':rejected,'sub_set':sub_set,'rejected2ids':rejected2ids,'chosen2ids':chosen2ids,'subsub_set':sub_sub_set}
    return dataset

def get_reward_bench2()->dict:
    """Return Dictionart with Reward benchmark 2.

    Returns:
    dict: Dictionart of Reward benchmark 2.
    """
    ids=[]
    chosen=[]
    rejected=[]
    sub_set=[]
    sub_sub_set=[]
    chosen2ids=[]
    rejected2ids=[]

    dataset=load_dataset('allenai/reward-bench-2')['test']
    for data in dataset:

        prompt=data['prompt']
        id=data['id']
        sub=data['subset']
        ids.append(id)
        sub_set.append(sub)
        sub_sub_set.append(sub)

        for c in data['chosen']:
            chosen.append(TEMPLET.format(prompt=prompt,assistant=c))
            chosen2ids.append(id)

        for r in data['rejected']:
            rejected.append(TEMPLET.format(prompt=prompt,assistant=r))
            rejected2ids.append(id)

    dataset={'ids':ids,'chosen':chosen,'rejected':rejected,'sub_set':sub_set,'rejected2ids':rejected2ids,'chosen2ids':chosen2ids,'subsub_set':sub_sub_set}
    return dataset

def load_rmb_dataset()->{}:
    """Loads RMB.

    Returns:
    dict: Return RMB.
    """
    rmb_dataset=[]

    paths2subsets={
        'datasets/RMB_dataset/BoN_set/Harmlessness/S6.json':'Privacy',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S10.json':'Sucide and Self-Harm',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S5.json':'Specialized Advice',
        'datasets/RMB_dataset/BoN_set/Harmlessness/multi.json':'Multi',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S1.json':'Violent Crimes',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S11.json':'Sexual Content',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S8.json':'Indiscriminate Weapons',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S3.json':'Sex-Related Crimes',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S9.json':'Hate',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S2.json':'Non-Violent Crimes',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S4.json':'Child Sexual Exploitation',
        'datasets/RMB_dataset/BoN_set/Harmlessness/S7.json':'Intellectual Property',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Generation/Professional Content Generation.json':'Generation',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Generation/Creative Writing.json':'Generation',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Translation/Literary and Cultural Translation.json':'Translation',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Translation/GeneralExcerpt Language Translation.json':'Translation',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Translation/Technical and Scientific Translation.json':'Translation',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Summarization/Standard Summaries.json':'Summarization',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Summarization/Specialized Summaries.json':'Summarization',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Code/Documentation.json':'Code',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Code/Quality and Optimization.json':'Code',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Code/Development and Implementation.json':'Code',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Code/Data Management.json':'Code',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Closed QA/ContextBased.json':'Closed QA',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Closed QA/OptionBased.json':'Closed QA',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Role Playing/General Character.json':'Role Playing',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Role Playing/Specific Character.json':'Role Playing',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Open QA/Hypothetical Scenarios.json':'Open QA',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Open QA/Interpretative Analysis.json':'Open QA',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Open QA/Personal Opinion and Advice.json':'Open QA',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Open QA/Factual.json':'Open QA',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Open QA/Technical and Practical Support.json':'Open QA',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Open QA/General Explanation.json':'Open QA',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Rewrite/PostQuality Assessment Rewriting.json':'Rewrite',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Rewrite/Tone Adjustment.json':'Rewrite',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Rewrite/Paraphrasing.json':'Rewrite',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Rewrite/Textual ExpansionReduction.json':'Rewrite',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Classification/Quality and Compliance Assessment.json':'Classification',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Classification/Content Categorization.json':'Classification',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Reasoning/Critical Thinking.json':'Reasoning',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Reasoning/Logical Deduction.json':'Reasoning',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Reasoning/Analytical Reasoning.json':'Reasoning',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Reasoning/Pattern Recognition.json':'Reasoning',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Reasoning/Human Decision Making.json':'Reasoning',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Chat/Supportive Conversation.json':'Chat',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Chat/Casual Conversation.json':'Chat',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Chat/Discussion.json':'Chat',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Brainstorming/Idea Development.json':'Brainstorming',
        'datasets/RMB_dataset/BoN_set/Helpfulness/Brainstorming/Problem Solving.json':'Brainstorming',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S6.json':'Privacy',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S10.json':'Sucide and Self-Harm',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S5.json':'Specialized Advice',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/multi.json':'Multi',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S1.json':'Violent Crimes',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S11.json':'Sexual Content',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S8.json':'Indiscriminate Weapons',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S3.json':'Sex-Related Crimes',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S9.json':'Hate',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S2.json':'Non-Violent Crimes',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S4.json':'Child Sexual Exploitation',
        'datasets/RMB_dataset/Pairwise_set/Harmlessness/S7.json':'Intellectual Property',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Generation/Professional Content Generation.json':'Generation',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Generation/Creative Writing.json':'Generation',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Translation/Literary and Cultural Translation.json':'Translation',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Translation/GeneralExcerpt Language Translation.json':'Translation',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Translation/Technical and Scientific Translation.json':'Translation',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Summarization/Standard Summaries.json':'Summarization',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Summarization/Specialized Summaries.json':'Summarization',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Code/Documentation.json':'Code',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Code/Quality and Optimization.json':'Code',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Code/Development and Implementation.json':'Code',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Code/Data Management.json':'Code',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Closed QA/ContextBased.json':'Closed QA',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Closed QA/OptionBased.json':'Closed QA',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Role Playing/General Character.json':'Role Playing',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Role Playing/Specific Character.json':'Role Playing',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Open QA/Hypothetical Scenarios.json':'Open QA',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Open QA/Interpretative Analysis.json':'Open QA',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Open QA/Personal Opinion and Advice.json':'Open QA',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Open QA/Factual.json':'Open QA',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Open QA/Technical and Practical Support.json':'Open QA',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Open QA/General Explanation.json':'Open QA',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Rewrite/PostQuality Assessment Rewriting.json':'Rewrite',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Rewrite/Tone Adjustment.json':'Rewrite',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Rewrite/Paraphrasing.json':'Rewrite',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Rewrite/Textual ExpansionReduction.json':'Rewrite',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Classification/Quality and Compliance Assessment.json':'Classification',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Classification/Content Categorization.json':'Classification',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Reasoning/Critical Thinking.json':'Reasoning',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Reasoning/Logical Deduction.json':'Reasoning',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Reasoning/Analytical Reasoning.json':'Reasoning',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Reasoning/Pattern Recognition.json':'Reasoning',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Reasoning/Human Decision Making.json':'Reasoning',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Chat/Supportive Conversation.json':'Chat',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Chat/Casual Conversation.json':'Chat',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Chat/Discussion.json':'Chat',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Brainstorming/Idea Development.json':'Brainstorming',
        'datasets/RMB_dataset/Pairwise_set/Helpfulness/Brainstorming/Problem Solving.json':'Brainstorming',
    }
    prompt_temp={
        "user":"Human: {p}\n\n",
        "assistant":"Assistant: {p}\n\n"
    }

    for path,sub_type in paths2subsets.items():

        subsub_set=path.split('/')[-1].replace('.json','')
        with open(path) as f:
            dataset = json.load(f)

        for data in dataset:
            prompt=''
            for input in data['conversation_input']:
                prompt+=prompt_temp[input['role']].format(p=input['content'])
            id=len(rmb_dataset)
            chosen=[]
            chosen_key = 'bon_best' if 'bon_best' in data else 'chosen'
            if type(data[chosen_key]) is list:
                for c in data[chosen_key]:
                    chosen.append("Assistant: "+c['answer'])
            else:
                chosen.append("Assistant: "+data[chosen_key]['answer'])

            loser_key = 'loser_list' if 'loser_list' in data else 'reject'
            rejected=[]
            if type(data[loser_key]) is list:
                for c in data[loser_key]:
                    rejected.append("Assistant: "+c['answer'])
            else:
                rejected.append("Assistant: "+data[loser_key]['answer'])

            rmb_dataset.append(
            {
                'id':id,
                'prompt':prompt,
                'subset':sub_type,
                'chosen':chosen,
                'rejected':rejected,
                'subsub_set':subsub_set
            }
            )
    return rmb_dataset

def get_rmb_dataset()->dict:
    """Return Dictionart with RMB.

    Returns:
    dict: Dictionart of RMB.
    """
    ids=[]
    chosen=[]
    rejected=[]
    sub_set=[]
    sub_sub_set=[]
    chosen2ids=[]
    rejected2ids=[]
    dataset=load_rmb_dataset()

    for data in dataset:

        prompt=data['prompt']
        id=data['id']
        sub=data['subset']
        ids.append(id)
        sub_set.append(sub)
        sub_sub_set.append(data['subsub_set'])

        for c in data['chosen']:
            chosen.append(prompt+c)
            chosen2ids.append(id)

        for r in data['rejected']:
            rejected.append(prompt+r)
            rejected2ids.append(id)
    dataset={'ids':ids,'chosen':chosen,'rejected':rejected,'sub_set':sub_set,'rejected2ids':rejected2ids,'chosen2ids':chosen2ids,'subsub_set':sub_sub_set}
    return dataset


def get_hh_rlhf_test()->dict:
    """Return Dictionart with hh rlhf helpful subset.

    Returns:
    dict: Dictionart of  hh rlhf helpful subset.
    """
    ids=[]
    chosen=[]
    rejected=[]
    sub_set=[]
    sub_sub_set=[]
    chosen2ids=[]
    rejected2ids=[]
    dataset_help = load_dataset("Anthropic/hh-rlhf",data_dir="helpful-base")['test']
    dataset_harm = load_dataset("Anthropic/hh-rlhf",data_dir="harmless-base")['test']
    id=0

    for data in dataset_help:
        sub='helpful'
        ids.append(id)
        sub_set.append(sub)
        sub_sub_set.append(sub)
        chosen.append(data['chosen'])
        chosen2ids.append(id)
        rejected.append(data['rejected'])
        rejected2ids.append(id)
        id=id+1

    for data in dataset_harm:
        sub='harmless'
        ids.append(id)
        sub_set.append(sub)
        sub_sub_set.append(sub)
        chosen.append(data['chosen'])
        chosen2ids.append(id)
        rejected.append(data['rejected'])
        rejected2ids.append(id)
        id=id+1
    dataset={'ids':ids,'chosen':chosen,'rejected':rejected,'sub_set':sub_set,'rejected2ids':rejected2ids,'chosen2ids':chosen2ids,'subsub_set':sub_sub_set}
    return dataset

def get_hh_rlhf_helpful_test()->dict:
    """Return Dictionart with hh rlhf helpful subset.

    Returns:
    dict: Dictionart of  hh rlhf helpful subset.
    """
    ids=[]
    chosen=[]
    rejected=[]
    sub_set=[]
    sub_sub_set=[]
    chosen2ids=[]
    rejected2ids=[]
    dataset_help = load_dataset("Anthropic/hh-rlhf",data_dir="helpful-base")['test']
    id=0

    for data in dataset_help:
        sub='helpful'
        ids.append(id)
        sub_set.append(sub)
        sub_sub_set.append(sub)
        chosen.append(data['chosen'])
        chosen2ids.append(id)
        rejected.append(data['rejected'])
        rejected2ids.append(id)
        id=id+1
    dataset={'ids':ids,'chosen':chosen,'rejected':rejected,'sub_set':sub_set,'rejected2ids':rejected2ids,'chosen2ids':chosen2ids,'subsub_set':sub_sub_set}
    return dataset

def get_hh_rlhf_harmless_test()->dict:
    """Return Dictionart with hh rlhf helpful subset.

    Returns:
    dict: Dictionart of  hh rlhf helpful subset.
    """
    ids=[]
    chosen=[]
    rejected=[]
    sub_set=[]
    sub_sub_set=[]
    chosen2ids=[]
    rejected2ids=[]
    dataset_harm = load_dataset("Anthropic/hh-rlhf",data_dir="harmless-base")['test']
    id=0

    for data in dataset_harm:
        sub='harmless'
        ids.append(id)
        sub_set.append(sub)
        sub_sub_set.append(sub)
        chosen.append(data['chosen'])
        chosen2ids.append(id)
        rejected.append(data['rejected'])
        rejected2ids.append(id)
        id=id+1
    dataset={'ids':ids,'chosen':chosen,'rejected':rejected,'sub_set':sub_set,'rejected2ids':rejected2ids,'chosen2ids':chosen2ids,'subsub_set':sub_sub_set}
    return dataset

def get_evaluation_benchmak(dataset_name:str)->dict:
    """Returns RM dataset."""
    print("Retriving datasets.")
    if dataset_name=='rm-bench':
        return get_rm_bench()
    elif dataset_name=='reward_bench':
        return get_reward_bench()
    elif dataset_name=='reward_bench2':
        return get_reward_bench2()
    elif dataset_name=='rmb_bench':
        return get_rmb_dataset()
    elif dataset_name=='hh-rlhf-helpful-harmless':
        return get_hh_rlhf_test()
    elif dataset_name=='hh-rlhf-harmless':
        return get_hh_rlhf_harmless_test()
    elif dataset_name=='hh-rlhf-helpful':
        return get_hh_rlhf_helpful_test()
    else:
        raise ValueError("Incorrect dataset name.")
