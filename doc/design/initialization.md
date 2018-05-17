# Initialization of Parameters

## Motivation

PaddlePaddle needs a way to init parameters.

## Challenge

Initialization operators must run once and only one; otherwise, every iteration would clear the parameters.

## Solution: Two seperate `ProgramDesc`.

The initialization part of the program in a `ProgramDesc` message, and the rest part in another.
