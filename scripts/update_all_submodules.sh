#!/bin/sh

# exec this script to update all submodules
git submodule init # Ensure the submodule points to the right place
git submodule sync    # Ensure the submodule points to the right place
git submodule update  # Update the submodule  
git submodule foreach git checkout master  # Ensure subs are on master branch
git submodule foreach git pull origin master # Pull the latest master
