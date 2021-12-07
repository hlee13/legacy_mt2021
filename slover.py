# Copyright 2010-2018 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal example to call the GLOP solver."""
# [START program]
# [START import]
from __future__ import print_function
from ortools.linear_solver import pywraplp
# [END import]

def create_variables(lb, ub, shape, name):
    lb, ub = p(lb), p(ub)
#         if len(shape) == 1 or shape[1] == 1:
#             return [solver.NumVar(lb, ub, '%s%d'%(name, i)) for i in range(shape[0])]
#         elif shape[0] == 1:
#             return [solver.NumVar(lb, ub, '%s%d'%(name, i)) for i in range(shape[1])]
#         else:
    # _r, _c = shape
    # return [[solver.NumVar(lb, ub, '%s%d_%d'%(name, i, j)) for j in range(_c)] for i in range(_r)]

#     [eval('%s = create_variables(lb, ub, shape, name)' %name) for lb, ub, shape, name in variables]
    
# x = [create_variables(lb, ub, shape, name) for lb, ub, shape, name in conf['variables']]
# x = [reduce(lambda _1, _2: _1+_2, xx) for xx in zip(*x)]

def dot(v1, v2):
  ans = 0
  for _1, _2 in zip(v1, v2):
    ans += _1 * _2
  return ans

def mut(M, v):
  ret = []
  for r in M:
    ret.append(dot(r, v))
  return ret

def multiply(M, A):
  ans = 0
  for i in range(len(M)):
    for j in range(len(M[0])):
      ans += M[i][j] * A[i][j]
  return ans

def flatten(_):
    return [e for ee in _ for e in ee]

def mock_engine(conf):
  solver = pywraplp.Solver(conf['name'] if 'name' in conf else 'name(default)', conf['engine'])
  inf = solver.infinity()
  def p(v):
      if v == 'inf':
          return inf
      if v == '-inf':
          return -inf
      return v
  
  # [START variables]
  x = [solver.NumVar(p(lb), p(ub), '%s%d'%(name, i)) for lb, ub, shape, name in conf['variables'] for i in range(shape)]
  print('Number of variables =', solver.NumVariables())
  # [END variables]
  
  # [START constraints]
  for _1, _2, _3 in zip(*eval(conf['s.t.'])):
      solver.Add(eval('_1 %s _3' %_2))
  print('Number of constraints =', solver.NumConstraints())
  # [END constraints]

  # [START objective]
  if 'maximize' in conf:
    solver.Maximize(eval(conf['maximize']))
  elif 'minimize' in conf:
    solver.Minimize(eval(conf['minimize']))
  else:
    raise Exception("['maximize', 'minimize'] not in conf")

  # [END objective]

  # [START solve]
  failed = solver.Solve()
  if failed:
    raise Exception('ret_code: ', failed)
  # [END solve]
  
  # [START print_solution]
  print('Solution:')
  print('Objective value =', solver.Objective().Value())

  ans = []
  for e in x:
      if isinstance(e, list):
          for _ in e:
            ans.append((_.name(), _.solution_value()))
      else:
          ans.append((e.name(), e.solution_value()))
  return ans

  # for e in x:
  #     if isinstance(e, list):
  #         for _ in e:
  #             print('%s=%f ' %(_.name(), _.solution_value()), end='')
  #         print('')
  #     else:
  #         print('%s=%f ' %(e.name(), e.solution_value()), )
  # [END print_solution]
A = [ [3, 2]
    , [2, 1]
    , [0, 3]
]
b = [65, 40, 75]
c = [1500, 2500]

A = [ [1,1,1,0,0,0,0,0,0]
    , [0,0,0,1,1,1,0,0,0]
    , [0,0,0,0,0,0,1,1,1]
    
    , [1,0,0,1,0,0,1,0,0]
    , [0,1,0,0,1,0,0,1,0]
    , [0,0,1,0,0,1,0,0,1]
    ]

b = [1] * 6
INF = 1e32
c = [[4, -INF, 3], [2, -INF, -INF], [-INF, 1, 5]]

def main():
  # primal
  ans = mock_engine(
        {
          'name': 'primal'
          , 'engine': pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
          , 'variables': [
              (0, 1, 3, 'x0')
              , (0, 1, 3, 'x1')
              , (0, 1, 3, 'x2')

              # (0, 'inf', len(c), 'x')
            ]
          , 'maximize': 'dot(flatten(c), x)'
          , 's.t.': 'mut(A, x), ["=="] * 6, b'
        }
    )
  for i in range(len(ans)):
    if i % 3 == 0:
      print()
    print(' %s: %7.3f'%(ans[i][0], ans[i][1]), end='')
  print()

  print('#' * 20)
  # dual
  ans = mock_engine(
        {
          'name': 'dual'
          , 'engine': pywraplp.Solver.GLOP_LINEAR_PROGRAMMING

          , 'variables': [
              ('-inf', 'inf', 3, 'x')
              , ('-inf', 'inf', 3, 'y')
              # (0, 'inf', len(b), 'y')
            ]
          , 'minimize': 'dot(b, x)'
          , 's.t.': 'mut(zip(*A), x), [">="] * 9, flatten(c)'
        }
    )
  for i in range(len(ans)):
    if i % 3 == 0:
      print()
    print(' %s: %7.3f'%(ans[i][0], ans[i][1]), end='')
  print()

if __name__ == '__main__':
  main()
# [END program]
