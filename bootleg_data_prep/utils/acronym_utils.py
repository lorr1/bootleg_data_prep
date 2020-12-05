'''
This file contains helper functions for identifying acronyms in text
'''

import string


def find_acronyms_from_parentheses(sentence_str, all_spans, all_qids, all_aliases):
    """
    Finds acronyms for (known) entities in a sentence, using information in parentheses
    
    Examples:
      If the original sentence in the document is: "The World Health Organization (WHO) is a specialized agency..." 
      then it will use the parentheses information to link WHO -> World Health Organization (after double-checking
      that WHO and "World Health Organization" match, using a greedy approach. See greedy_match())
      
      It also allows for a more lenient form of greedy matching that doesn't require an exact match. e.g. if the
      sentence is "The Center for Disease Control and Prevention (CDC) is a national ..." then it will still
      identify CDC as a match for "The Center for Disease Control and Prevention" despite the other words such
      as "The", "for", "and", "Prevention"
      
    Parameters:
      sentence_str: list of words from the sentence
      all_spans: list of spans of named entities in the sentence
      all_qids: list of qids of named entities in the sentence
      all_aliases: list of aliases of named entities in the sentence
    
    Returns:
      dictionary of acronym -> tuple(qid, alias)
      where alias is the original alias that was connected to the named entity
    """
    # Firstly, if all_spans or all_qids are empty, there's no point in searching for acronyms because we 
    # won't have any qids to link them to
    if len(all_spans) == 0 or len(all_qids) == 0:
        return {}

    # Simple checks to make sure nothing weird is going on with parentheses in the input sentence
    if sentence_str.count('(') != sentence_str.count(')'): # unbalanced parentheses
        return {}
    parens_only = [x for x in sentence_str if x in ['(', ')']]
    if not check_alternating_parentheses(parens_only): # check that parentheses are balanced
        return {}
    
    left = 0
    right = 0
    acronym_to_qid = {}
    while left < len(sentence_str) and right < len(sentence_str):
        if sentence_str[left] == "(":
            while sentence_str[right] != ")":
                right += 1
                if right == len(sentence_str): # reached end of sentence, so we should exit the function
                    return acronym_to_qid
            # If the code gets here, then we have successfully found a pair of parentheses
            words = sentence_str[left+1:right]
            # Only continue under two conditions: firstly, if the entire contents of the parentheses is a single word that is at least
            # 3 letters, because acronyms should be 1 word and at least 3 letters; or secondly, if the parentheses are something like 
            # "The World Health Organization (WHO, established in ...)" and you can extract the acronym by splitting on the comma.
            # First, let's find whether we are in the second scenario or not. If yes, comma_flag will be set to True.
            comma_flag = False
            if words.count(',') == 1:
                comma_idx = words.index(',')
                words_split_on_comma = [words[:comma_idx], words[comma_idx+1:]]
                # only keep the acronym if it's the only 1-word part in the parentheses; if multiple, it's a bit risky
                if len(words_split_on_comma[0]) == 1 and len(words_split_on_comma[1]) == 1:
                    comma_flag = False
                else:
                    for w in words_split_on_comma:
                        if len(w) == 1:
                            if len(w[0]) >= 3:
                                acronym = w[0]
                                comma_flag = True

            if (len(words) == 1 and len(words[0]) >= 3) or comma_flag:
                if not comma_flag:
                    acronym = words[0] # if comma_flag == True, then acronym was already set above
                # Make sure word is at least 2/3 upper case letters, otherwise it's probably not an acronym
                if sum(map(str.isupper, acronym)) / len(acronym) >= 2/3:
                    # Now check if there is a labeled mention immediately to the left of the current set of parentheses
                    entity = None
                    entity_qid = None
                    entity_alias = None
                    for idx, span in enumerate(all_spans):
                        l,r = span
                        if r == left:  # mention should be immediately to the left of the left parenthesis
                            entity = sentence_str[l:r]
                            entity_qid = all_qids[idx]
                            entity_alias = all_aliases[idx]
                            break
                    if entity is not None:
                        matches = greedy_match(entity, acronym)
                        if matches:
                            acronym_to_qid[acronym] = (entity_qid, entity_alias)
            left = right + 1 # whether or not we found a match above, we want to jump left ahead to get past the current set of parentheses
            right = left
        else:
            left += 1
            right = left
    return acronym_to_qid

def check_alternating_parentheses(alist):
    """
    Returns True if the elements of alist are parentheses that alternate correctly, starting with a left paren

    Examples:
      ['(', ')', '(', ')']  --> True
      [')', '(', ')', '(']  --> False
      ['(', '(']            --> False
    """
    for i in range(0, len(alist), 2):
        if alist[i] != "(":
            return False
    for i in range(1, len(alist), 2):
        if alist[i] != ")":
            return False
    return True

def greedy_match(entity, acronym):
    """
    Returns True if there is a match between the entity and acronym, False if not

    Example: "Center for Disease Control and Prevention" matches "CDC" because of the words "Center", "Disease", and "Control"
    """
    # For greedy matching, we will NOT require that the match be case sensitive
    acronym = acronym.lower()
    entity = [x.lower() for x in entity]

    pointer = 0
    for word in entity:
        if word[0] == acronym[pointer]:
            pointer += 1
        if pointer == len(acronym):
            return True
    return False

def find_manufactured_acronyms(sentence_str, all_spans, all_qids, all_aliases):
    """
    Finds POSSIBLE acronyms for (known) entities in a sentence, by "manufacturing" acronyms, i.e. by taking the first
    letter of each word in the entity and combining them into an acronym. This code does it three ways:
    1. by taking the first letter of each word, 2. by taking the first letter of each word while ignoring common
    words like "the" and "of," 3. by taking the first letter of each word and capitalizing it
    
    Examples:
      "Organization of the Petroleum Exporting Countries" -> this code would extract "OotPEC," "OPEC," and "OOTPEC"
      
    Parameters:
      sentence_str: list of words from the sentence
      all_spans: list of spans of named entities in the sentence
      all_qids: list of qids of named entities in the sentence
      all_aliases: list of aliases of named entities in the sentence
    
    Returns:
      dictionary that maps acronym -> tuple(qid, alias)
      where alias is the original alias that was connected to the named entity
    """
    manufactured_acronyms = {}
    for span_idx, span in enumerate(all_spans): # get entity mentions from spans/sentence, not from sentence['aliases']
        start, end = span
        named_entity = sentence_str[start:end]
        named_entity_qid = all_qids[span_idx]
        named_entity_alias = all_aliases[span_idx]

        # Try three versions of acronyms. First, just take the first letter of each word. Secondly, try removing common 
        # prepositions/conjunctions etc. that are often ignored when making acronyms. e.g., for "Organization of the Petroleum 
        # Exporting Countries" we would try both "OotPEC" and also "OPEC" (the 2nd one is correct). Thirdly, take the first letter
        # of each word, but make everything upper case (e.g. "Department of Defense" -> "DOD" instead of "DoD").
        potential_acronyms = []
        ignore_words = set(["The", "the", "Of", "of", "And", "and", "For", "for", "In", "in",
                            "To", "to"]) # maybe also try adding "with", "a", "on", etc. ??
        potential_acronyms.append(''.join([word[0] for word in named_entity]))
        potential_acronyms.append(''.join([word[0] for word in named_entity if word not in ignore_words]))
        potential_acronyms.append(''.join([word[0].upper() for word in named_entity]))
        for potential_acronym in potential_acronyms:
            # Only consider words that are at least 3 letters long. With 1 or 2 letters, it's too easy
            # for an acronym to match some entity purely by chance
            if len(potential_acronym) < 3:
                continue
            # If the entity has any words that start with numbers or punctuation marks, skip that entity
            # (we don't want to have any acronyms like "4SB" or "S,I")
            if any(word[0] in string.punctuation for word in named_entity):
                continue
            if any(str.isdigit(word[0]) for word in named_entity):
                continue
            # Make sure at least 2/3 of the letters are upper-case. Basically, we are using upper-case info to 
            # tell whether the word is an acronym or not. 
            # Specifically, if at least 2/3 of the letters are upper case, we will consider it
            # to be an acronym. The idea behind this is that most acronyms are upper case; e.g. "WHO" is
            # probably an acronym but "who" is probably not. However, some acronyms have a few lower case 
            # letters, such as "IoT", so we arbitrarily set the threshold at 2/3 to be a bit more lenient
            # (the 2/3 number could be changed in the future).
            if sum(map(str.isupper, potential_acronym)) / len(potential_acronym) < 2/3:
                continue
            # If it passes all the checks above, then add to the dictionary as a possible acronym
            manufactured_acronyms[potential_acronym] = (named_entity_qid, named_entity_alias)
    return manufactured_acronyms

def get_span_of_list_in_list(list1, list2):
    """
    Checks if the contents of list1 are also inside list2, sequentially. If not, returns None. If yes,
    returns the span of list2 that contains the contents of list1

    Examples:
      get_span_of_list_in_list(["Testing", "123"], ["Testing", "this", "123"]) returns None
      get_span_of_list_in_list(["Testing", "123"], ["Test1", "Testing", "123"]) returns [1,3]

    Adapted from https://stackoverflow.com/a/3847585
    Note the somewhat unusual usage of the else clause on the for loop
    """
    for i in range(len(list2)-len(list1)+1):
        for j in range(len(list1)):
            if list2[i+j] != list1[j]:
                break
        else:
            return [i, i+len(list1)]
    return None

def augment_first_sentence(sent_aliases, sent_spans, sent_qids, title, sentence_str, args, qid2singlealias, qid2alias, doc_entity):
    '''
    The first sentence of Wikipedia articles often repeat the title/main entity of the page, but it isn't connected
    to an anchor link (e.g. "The World Health Organization (WHO) is..."). But these are useful for acronym mining.
    So, for the first sentence, we add a link for the main entity IF it appears in the sentence.

    So this function takes in the original lists of sentence aliases/spans/qids for the first sentence of the article,
    and if the title appears in the first sentence, we add it to those lists. This will be used later in the acronym pipeline.

    NOTE: this may seem slightly redundant since the mention will likely be picked up later in the weak labeling pipeline, 
    but I'm still adding it because this occurs BEFORE acronym mining starts, whereas other weak labeling occurs 
    AFTER acronym mining starts.
    '''
    sentence_aliases_augmented = sent_aliases.copy()
    sentence_spans_augmented = sent_spans.copy()
    sentence_qids_augmented = sent_qids.copy()

    # Check if the title of the document appears in the first sentence
    temp_span = get_span_of_list_in_list(title.split(), sentence_str) # returns span of doc_entity in sentence, or None if not in sentence
    if temp_span is not None:
        if not args.no_permute_alias: # use most conflicting alias
            sentence_aliases_augmented.append(qid2singlealias.get(doc_entity))
        else:
            if title.lower() in qid2alias.get(doc_entity): # try using the title (but only if it's already in the alias table for the CORRECT qid)
                sentence_aliases_augmented.append(title.lower())
            else: # otherwise use the most conflicting alias
                sentence_aliases_augmented.append(qid2singlealias.get(doc_entity))
        sentence_spans_augmented.append(temp_span)
        sentence_qids_augmented.append(doc_entity)
        
        # Need to sort the 3 lists above, because by appending to the end, we didn't necessarily add doc_entity in the 
        # right location in the list. We zip them together so that we can sort all 3 lists based on sentence_spans_augmented
        if len(sentence_spans_augmented) > 1:
            zipped = list(zip(sentence_spans_augmented, sentence_aliases_augmented, sentence_qids_augmented))
            zipped.sort(key = lambda x: x[0][0])
            sentence_spans_augmented, sentence_aliases_augmented, sentence_qids_augmented = [list(x) for x in zip(*zipped)] # unzip tuples to lists

    return sentence_aliases_augmented, sentence_spans_augmented, sentence_qids_augmented