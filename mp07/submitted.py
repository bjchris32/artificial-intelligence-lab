'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def something_in_list(list):
    if ('something' in list):
        return True
    return False

def something_in_list_of_list(list_of_list):
    for list in list_of_list:
        if something_in_list(list):
            return True
    return False

def get_recursive_value(dic, key):
    # print("dic = ", dic)
    # print("key = ", key)
    if key in dic:
        temp_value = dic[key]
        recursive_value = get_recursive_value(dic, temp_value)
        if recursive_value == None:
            return temp_value
        else:
            return recursive_value
    else:
        return None

# the consequent of a rule that has an empty antecedents list
def goal_set_all_true(rules, goal_set):
    # print("goal_set_all_true")
    # print("goal_set_all_true -------- rules = ", rules)
    # print("goal_set_all_true -------- goal_set = ", goal_set)
    goal_set_truth = [False * len(goal_set)]
    for idx, goal in enumerate(goal_set):
        for rule in rules:
            if goal == rule['consequent']:
                # check antecedents are empty
                antecedents_all_empty = True
                for antecedent in rule['antecedents']:
                    if antecedent != []:
                        antecedents_all_empty = False
                if antecedents_all_empty:
                    goal_set_truth[idx] = True
    # print("goal_set_truth = ", goal_set_truth)
    return all(goal_set_truth)

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    standardized_rules = {}
    variables = []

    for rule_key in nonstandard_rules.keys():
        standardized_rules[rule_key] = {}
        var_prefix = rule_key + 'var'
        var_counter = 0
        rule = nonstandard_rules[rule_key]
        antecedents = rule['antecedents']
        consequent = rule['consequent']
        text = rule['text']

        antecedents_replaced = antecedents
        consequent_replaced = consequent
        if len(antecedents) > 0 or len(consequent) > 0:
            # TODO: flatten the array to check
            if something_in_list_of_list(antecedents) or something_in_list(consequent):
                var_counter += 1
                temp_var = var_prefix + str(var_counter)
                variables.append(temp_var)
                # replace something in the rule with temp_var
                antecedents_replaced = []
                for antecedent in antecedents:
                    temp_antecedent = []
                    for s in antecedent:
                        if s == 'something':
                            temp_antecedent.append(temp_var)
                        else:
                            temp_antecedent.append(s)
                    antecedents_replaced.append(temp_antecedent)
                
                temp_consequent = []
                for s in consequent:
                    if s == 'something':
                        temp_consequent.append(temp_var)
                    else:
                        temp_consequent.append(s)
                consequent_replaced = temp_consequent

        standardized_rules[rule_key]['antecedents'] = antecedents_replaced
        standardized_rules[rule_key]['consequent'] = consequent_replaced
        standardized_rules[rule_key]['text'] = text
        
    return standardized_rules, variables

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    unification = None
    subs = None
    query_bool = query[-1]
    datum_bool = datum[-1]
    if query_bool != datum_bool:
        return unification, subs

    subs = {}
    unification = copy.deepcopy(query)
    datum_copy = copy.deepcopy(datum)

    i = 0
    while(i < len(unification)):
        if unification[i] in variables:
            subs[unification[i]] = datum_copy[i]
            # replace query with sub
            replacing_indexes = [k for k in range(len(unification)) if unification[k] == unification[i]]
            replace_var = subs[unification[i]]
            for k in replacing_indexes:
                unification[k] = replace_var

        elif datum_copy[i] in variables:
            subs[datum_copy[i]] = unification[i]
            # replace query with sub
            replacing_indexes = [k for k in range(len(unification)) if unification[k] == datum_copy[i]]
            replace_var = subs[datum_copy[i]]
            for k in replacing_indexes:
                unification[k] = replace_var
        i += 1

    return unification, subs

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    applications = []
    goalsets = []
    # print("rule['consequent'] = ", rule['consequent'])
    for idx, goal in enumerate(goals):
        # 1. Test to see whether the consequent of the rule can be unified with a goal
        unification, subs = unify(rule['consequent'], goal, variables)
        # print("unify the goal: ", goal)
        # print("unification =", unification, " subs = ", subs)

        if unification == None and subs == None:
            continue
        # unify the goal:  ['bobcat', 'eats', 'squirrel', False]
        # unification = ['bobcat', 'eats', 'squirrel', False]  subs =  {'x': 'bobcat'}

        # 2. Take the variable substitutions from the rule application,
        #    and modify the rule antecedents using those same substitutions.
        temp_antecedents = copy.deepcopy(rule['antecedents'])
        for temp_antecedent in temp_antecedents:
            antecedent_index = 0
            while(antecedent_index < len(temp_antecedent)):
                substitue = get_recursive_value(subs, temp_antecedent[antecedent_index])
                if substitue is not None:
                    temp_antecedent[antecedent_index] = substitue
                antecedent_index += 1

        applications.append({
            'antecedents': temp_antecedents,
            'consequent': unification
        })

        # 3. Replace the old goal with the new goals
        # the goal that unified with
        # applications[i]['consequent'] has been removed, and replaced by
        # the members of applications[i]['antecedents']
        temp_goals = copy.deepcopy(goals)
        temp_goals.pop(idx)
        temp_goals = temp_goals + temp_antecedents
        goalsets.append(temp_goals)

    return applications, goalsets

def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    # raise RuntimeError("You need to write this part!")
    proof = []

    # print("query = ", query)
    # print("rules = ", rules)
    # print("variables = ", variables)

    # The starting state is the statement that you are trying to prove.
    # Every action is a rule application. The result of an action is to create a new goalset.
    # Every state is a goalset. A goalset is a list of propositions called "goals" such that, if every goal is proven true, then that constitutes a proof of the statement you are trying to prove.
    # The ending state is an empty goalset. The ending state is achieved by transforming the starting state into a list of goals such that every goal in that list is a known true proposition,
    #  i.e., the consequent of a rule that has an empty antecedents list.

    # Slide explaination:
    # starting state: goal set
    # actions: possible set of rules
    # neighboring states: if Q unifies with Q' in goal set
      # replace Q' with the unified antecedents
    # terminate the goal set contains only truth

    # functions defined:
    # unify(query, datum, variables)
    # return unification, subs

    # apply(rule, goals, variables)
    # return applications, goalsets
    proof = None

    queue = []
    queue.append([rules.values(), [query]])
    # BFS: iterate rules and try to unify the rule to generate new goal set
      # check if the goal set contains only truth.
      # i.e. the consequent of a rule that has an empty antecedents list.
    while queue:
        current_rules, current_goal_set = queue.pop(0)
        # print("######### current_rules = ", current_rules)
        # print("######### current_goal_set = ", current_goal_set)
        
        # Q: How to terminate the search?
        # Sol1: Check antecedent or consequent?
        # Sol2: empty goal set?
        # check if every goal in current_goal_set is a known true proposition
        # (the consequent of a rule that has an empty antecedents list)
        if len(current_goal_set) == 1 and len(current_goal_set[0]) == 0:
            return True
        # Is it necessary?
        # if goal_set_all_true(current_rules, current_goal_set):
        #     return True

        # get all neighbors with current_goal_set 
        for rule in current_rules:
            # print("rule = ",  rule)
            applications, goalsets = apply(rule, current_goal_set, variables)
            # Q: What should be pushed intto queue? goalsets or applications?
            # -> Both
            # print("applications = ", applications, "goalsets = ", goalsets)
            # if len(goalsets) == 1 and len(goalsets[0]) == 0:
            #     continue
            queue.append([applications, goalsets])
            # print("queue == ", queue)
            # break
        # print("queue == ", queue)
    return proof
