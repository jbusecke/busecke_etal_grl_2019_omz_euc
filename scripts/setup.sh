#!/bin/bash
# Create a github_repo from the command line...so cool (more here: https://developer.github.com/v3/repos/#create)
# I should probably implement some check if the creation was successful
curl -u 'jbusecke' https://api.github.com/user/repos -d '{"name":"busecke_etal_grl_2019_omz_euc", "private":"false"}'

# Link local repository to git
git init
git add *
git add .gitignore .stickler.yml .travis.yml
git commit -m 'first commit'
git remote add origin git@github.com:jbusecke/busecke_etal_grl_2019_omz_euc.git
git push -u origin master
