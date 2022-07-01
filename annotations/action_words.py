from nltk.corpus import framenet as fn
import re

# Set of abstract frames that correspond to actions
action_seed_frames = {
    'Transitive_action',  # Agent or Cause affecting a Patient
    'Intentionally_act',  # Acts performed by sentient beings
    'Getting',  # 'I got two whistles from John.'
    'Go_into_shape',  # Non-agentive shape manipulation
    'Misdeed',  # A Wrongdoer engages in a Misdeed
    'Traversing',  # Theme changes location with respect to a salient location
    'Waking_up'  # Sleeper transitions from a state of consciousness
}


def clean_parenthesis(name):
    name = name.split('_')[0]
    # remove words within ()
    name = re.sub(r"\s*\(+[^()]*\)+\s*", " ", name)
    # remove words within []
    name = re.sub(r"\s*\[+[^()]*\]+\s*", " ", name).strip()
    # remove extra spaces
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_lexeme_fn(lex_unit):
    """
    Get the lexeme in the form  of 'lemma.POS' from a FrameNet LexicalUnit

    Parameters
    ----------
    lex_unit: fn.LexicalUnit

    Returns
    -------
    str
        lemma.POS
    """
    head_lexeme = lex_unit.lexemes[0]
    if len(lex_unit.lexemes) > 1:
        for lexeme in lex_unit.lexemes:
            if lexeme.headword == 'true':
                head_lexeme = lexeme
                break
    return clean_parenthesis(head_lexeme.name) + '.' + lex_unit.POS.lower()


def get_derived_frames(seed_frames):
    """
    Go through the FrameNet inheritance hierarchy starting with a
    list of seed frames and generate a list of frames

    Parameters
    ----------
    seed_frames: set[str]
        A set of strings of frame names
    Returns
    -------
    set[str]
        A set of strings of derived frame names
    """
    # maintain a que of derived frames from the seed
    frame_que = set(seed_frames)

    # set to store the identified derived frames
    derived_frames = set()

    # go through added frames in the que that follow the inheritance relations
    while len(frame_que) != 0:
        f_name = frame_que.pop()
        derived_frames.add(f_name)
        frame = fn.frame_by_name(f_name)
        # relation is of the form (superFrame --> relation_type --> subFrame)
        for relation in frame.frameRelations:
            super_f_name = relation.superFrameName
            if super_f_name == f_name:
                sub_f_name = relation.subFrameName
                relation_type = relation.type.name
                # relation: (superFrame --> 'Inheritance' --> subFrame)
                # all children of Communication are actions even though not Inheritance
                if super_f_name == 'Communication' or relation_type == 'Inheritance':
                    if sub_f_name not in derived_frames:
                        frame_que.add(sub_f_name)

    return derived_frames


def action_filter(lex_unit):
    """
    Filter out non-action lexical units

    Parameters
    ----------
    lex_unit: dict
        A dict of LexicalUnit

    Returns
    -------
    bool
        True if it's a valid action lex unit
    """
    # if the lex unit is a verb, return True
    if lex_unit.name.endswith('.v'):
        return True
    elif lex_unit.name.endswith('.n'):
        agent_nouns = {'one', 'who', 'person', 'entity', 'someone', 'somebody'}

        # some lex unit names have information like [entity], [person] etc.
        entity_names = set([f'[{word}]' for word in ['entity', 'item', 'person']])

        agentive_sem_types = {'Agentive_noun', 'Artifact', 'Participating_entity',
                              'Physical_entity', 'Physical_object'}

        # check if for 5 words in the definition of lex unit has an agent noun
        # presence of those words point to the lex unit being a role instead of action
        agent_in_def = len(agent_nouns.intersection(lex_unit.definition.split()[:3])) > 0

        # semType contains agentive sem type
        has_agentive_sem = len(agentive_sem_types.intersection([s.name for s in lex_unit.semTypes])) > 0

        # has entity name in lex unit name
        has_entity_in_name = sum([name in lex_unit.name for name in entity_names]) > 0

        # ! (A or B or C) == !A and !B and !C
        return not (agent_in_def or has_agentive_sem or has_entity_in_name)
    else:    # adjective/adverbs/etc
        return False


def get_action_lexical_units_fn(action_seeds):
    """
    Go through the FrameNet inheritance hierarchy starting with a list
    of seed frames and generate a list of lexical units that correspond
    to action following the inheritance from the seed frames

    Parameters
    ----------
    action_seeds: set[str]
        A list of FrameNet frame names

    Returns
    -------
    list
        A list of lexical units of the form [(lexeme, frame_name, lex_unit_name)]
    """
    # get frames derived from the action seed frames
    action_derived_frames = get_derived_frames(action_seeds)

    lu_list = []

    for f_name in action_derived_frames:
        frame = fn.frame_by_name(f_name)
        for lex_unit in frame.lexUnit.values():
            if action_filter(lex_unit):
                lu_list.append((get_lexeme_fn(lex_unit), f_name, lex_unit.name))

    return lu_list


if __name__ == '__main__':
    lus = get_action_lexical_units_fn(action_seed_frames)
    lus = sorted(lus, key=lambda x: (x[0][-1], x[0], x[1]))

    with open('lexical_units_actions.tsv', 'w') as alf:
        alf.write('lexeme\tframe_name\tlexUnit_name\n')
        alf.write('\n'.join(['\t'.join(lu) for lu in lus]))
