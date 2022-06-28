from pulp import *
from random import randint


def n_choose_2(n):
    """
    All combinations of pairs of unique integers between 0 and n

    Parameters
    ----------
    n: int

    Returns
    -------
    int, int
    """
    for i in range(n):
        for j in range(i):
            yield i, j


def trigger_frame_score(triggers, candidates):
    """
    Generate a score for trigger and all its candidates
    Example scores would be edit distance between the trigger and candidate title,
    tfidf cosine similarity/BM25 between the tokens of the doc and the tokens of the candidate KB description, etc

    Parameters
    ----------
    triggers: list
    candidates: list

    Returns
    -------
    dict
    """
    all_scores = {}

    for i, trigger in enumerate(triggers):

        for k, candidate in enumerate(candidates[i]):
            all_scores[(i, k)] = randint(-100, 100) / 100

    return all_scores


def frame_frame_score(triggers, candidates):
    """
    Generate a score between pair of KB entries.
    Examples would be: overlap between the inlinks or outlinks of the KB entries,
    tfidf cosine similarity/BM25 score between the tokens of the descriptions of the KB entries, etc

    Parameters
    ----------
    triggers: list
    candidates: list

    Returns
    -------
    dict
    """
    n = len(triggers)

    all_scores = {}

    for i, j in n_choose_2(n):
        trigger_i, trigger_j = triggers[i], triggers[j]
        for k, cand_k in enumerate(candidates[i]):
            for l, cand_l in enumerate(candidates[j]):
                all_scores[(i, j, k, l)] = randint(-500, 100) / 100

    return all_scores


def optimal_coherence(triggers,
                      candidates,
                      nlp_doc,
                      trigger_kb_score=trigger_frame_score,
                      kb_kb_score=frame_frame_score):
    """
    Finds the optimal set of candidates associated to the triggers in the document based on how coherent
    they are to appear together in the KB
    ILP adopted from: Relational Inference for Wikification: https://aclanthology.org/D13-1184.pdf

    Parameters
    ----------
    triggers: list of spacy.tokens.Span
    candidates: list of list of Knowledge Base (KB) entries of each trigger respectively
    nlp_doc: spacy.tokens.Doc
    trigger_kb_score: func to find the similarity between a trigger and KB entry
    kb_kb_score: func to find the similarity between a pair of KB entries

    Returns
    -------
    list
        The list of candidates associated to the trigger that give the optimal coherence.
        Each element in the list is either one of the candidates of the trigger or None
    """

    problem = LpProblem("Maximize Coherence", LpMaximize)

    # create index lists for the triggers and candidates
    n = len(triggers)  # number of triggers
    i_s = list(range(n))  # trigger indices
    j_s = list(range(n))  # trigger indices
    k_s = list([list(range(len(candidates[i]))) for i in i_s])  # candidate indices
    l_s = list([list(range(len(candidates[j]))) for j in j_s])  # candidate indices

    # generate the score matrix for trigger and its candidates
    trigger_kb_scores = trigger_kb_score(triggers, candidates)

    # generate the coherence score for kb and kb entries
    kb_kb_scores = kb_kb_score(triggers, candidates)

    # create the 0,1 variables for the candidates of the triggers
    e = LpVariable.dicts("e_ik", ((i, k) for i in i_s for k in k_s[i]), cat="Binary")

    # create the pair-wise 0,1 variables for each candidate (k) for each trigger (i)
    # with the candidate (l) of another trigger (j)
    r = LpVariable.dicts("r_ij_kl",
                         ((i, j, k, l) for i, j in n_choose_2(n) for k in k_s[i] for l in l_s[j]),
                         cat="Binary")

    # objective function:
    # # Local part scores the trigger and its kb candidates
    local_part = lpSum(e[i, k] * trigger_kb_scores[i, k] for i in i_s for k in k_s[i])
    # # Global part scores the coherence of the chosen candidates of all the triggers in the KB
    global_part = lpSum(r[i, j, k, l] * kb_kb_scores[i, j, k, l]
                        for i, j in n_choose_2(n) for k in k_s[i] for l in l_s[j])
    problem += local_part + global_part, "Objective function"

    # unique or None solution constraint for the candidates of triggers
    # for solutions without None/NIL candidates, change '<=' to '=='
    # we could also add None as one of the candidates and assign a score of 0. to it
    for i in i_s:
        problem += lpSum(e[i, k] for k in k_s[i]) <= 1

    # relation constraint: i.e., the instance when the candidate of i is k "and" candidate of j is l
    # 2 * r_ij_kl <= e_ik + e_jl
    for i, j in n_choose_2(n):
        for k in k_s[i]:
            for l in l_s[j]:
                problem += 2 * r[i, j, k, l] <= 1 * e[i, k] + 1 * e[j, l]

    ### Solve the ILP ###
    problem.solve(PULP_CBC_CMD(msg=0))

    # generate the optimal candidates list
    optimal_candidates = ([None] * n)[:]
    for i in i_s:
        for k, candidate_k in enumerate(candidates[i]):
            if e[i, k].value() > 0.:
                optimal_candidates[i] = candidate_k
            # print(e[i, k].name, e[i, k].value())

    # print('Status:', LpStatus[problem.status])
    return optimal_candidates


if __name__ == '__main__':
    triggers_test = [1, 2, 3]
    candidates_test = [[1, 2], [3, 4, 5], [6]]
    opt_cands = optimal_coherence(triggers_test, candidates_test, None)
    print(opt_cands)