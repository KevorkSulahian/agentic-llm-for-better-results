# Crew - Multi-agent System

This page explains the general aspects of a crew of LLM agents, and particularly with
respect to financial tasks.

For more information about the specific crews in the FinMAS app, [click here](../crews/index.md).

## Configuration

The configuration of a crews consists of defining the agents, the tasks and which
tools each agent can use to solve their assigned tasks.

The performance of a configuration of agents and tasks depend a lot on which LLM model
is used for the agent executing the task. For example an Agent configuration can work
very well for `gpt-4o-mini` while for `llama-3.2-8b` there is likely another configuration
that will work better.

## Agents

According to the CrewAI framework, [agents](https://docs.crewai.com/concepts/agents) are defined with a name, role, and a backstory.
This gives the agents a pre-defined setup that guides the agents in how they should
go about solving the tasks they are assigned.

It is important that the wording of the configuration is clear and concise, as each
word will be given to the agent and thus uses up space in the context window for the LLM.
Therefore, it is important to avoid writing general terms and vague wording.

## Tasks

A [task](https://docs.crewai.com/concepts/tasks) is defined by a name, a description, and an expected output format. As with the agent
configuration, it is important to be clear and concise in the wording as each word will
be given to the LLM, and slight variations in the wording can lead to different results.

It is important to define in detail what the expected output format and what the content
should be.

## Tools

The agents can use assigned [tools](https://docs.crewai.com/concepts/tools) to solve their tasks.
The tools in this project consists of fetching data from a dedicated data source. The tools
include preprocessing of the data so that an LLM can better extract the information needed
to solve the task.
