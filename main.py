from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Creola Katherine Johnson (née Coleman; August 26, 1918 – February 24, 2020) was an American human computer whose calculations of orbital mechanics as a NASA employee were critical to the success of the first and subsequent U.S. crewed spaceflights.[1][2] During her 33-year career at NASA and its predecessor, the National Advisory Committee for Aeronautics, she earned a reputation for mastering complex manual calculations and helped pioneer the use of computers to perform tasks previously requiring humans. The space agency noted her "historical role as one of the first African-American women to work as a NASA scientist".[3]

Johnson's work included calculating trajectories, launch windows, and emergency return paths for Project Mercury spaceflights, including those for astronauts Alan Shepard, the first American in space, and John Glenn, the first American in orbit, and rendezvous paths for the Apollo Lunar Module and command module on flights to the Moon.[4] Her calculations were also essential to the beginning of the Space Shuttle program, and she worked on plans for a human mission to Mars.

In 2015, President Barack Obama awarded Johnson the Presidential Medal of Freedom. In 2016, she was presented with the Silver Snoopy Award by NASA astronaut Leland D. Melvin and a NASA Group Achievement Award. She was portrayed by Taraji P. Henson as a lead character in the 2016 film Hidden Figures. In 2019, Johnson was awarded the Congressional Gold Medal by the United States Congress.[5] In 2021, she was inducted posthumously into the National Women's Hall of Fame.[6]
    """

    summary_template = """
    Given the information {information} about a person, I want you to create: 
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    chain = summary_prompt_template | llm

    result = chain.invoke({"information": information})
    print(result.content)


if __name__ == "__main__":
    main()
