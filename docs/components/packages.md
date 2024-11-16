# Packages

The following sections include a short description about key packages that
are used in this project. We highlight why and how these packages are used
in the project.

## crewAI

[crewAI](https://github.com/crewAIInc/crewAI) is a multi-agent orchestration package that provides a framework for how
a system (or a crew of agents) can be defined and executed. The main components
of the framework are Agents, Tasks, Tools, and Crew.

Each Agent is configured with a Role, Goal and Backstory that guides the agent
to how it will approach and solve Tasks that are given to it.
Tasks are defined with a Description and Desired Output.

A essential part of an autonoumous agent is the ability to use specified tools
to solve the tasks. In the FinMAS system the Agents are given tools to fetch
news data and SEC filing data.

The Agents, Tasks, and Tools are then put together in a Crew than can receive
some input data and will work to solve the tasks given to them.

## llama-index

A powerful use-case for LLMs is when the LLMs are fed custom data to solve
their task. By doing this it is possible to adapt an LLM to solve unique tasks
that would otherwise not be possible for a general-purpose LLM that only have
access to the data that it was trained on.

[llama-index](https://docs.llamaindex.ai/) is a package that provides a framework for developing RAG applications
where the goal is to make it easy for the user to import custom data and transform
that data into a Vector Store by using an embedding model. After the data is transformed,
the framework provides capabilities for how to connect the Vector Store
with LLMs so that it is possible to query or have a chat with the data.

## panel

Developing a user interface for any app with a lot of data and configuration can be
quite time consuming. Therefore, we use the [panel](https://panel.holoviz.org/) package, which is a
web app framework focused on data science applications. There are numeruous web frameworks
available in Python, however, panel provides some unique features in terms of
advanced data visualization and interactivity that makes it favorable
to use for our project where we want to present information on multiple tabs
and in an interactive manner.
