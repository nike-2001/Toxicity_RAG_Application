import streamlit as st
import os
import logging
from typing import List, Dict, Any
import json
import time
import re

# Import necessary libraries
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Setup
logging.basicConfig(level=logging.INFO)

# Page config
st.set_page_config(
    page_title="🛡️ Toxicity Detection & Content Moderation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = None
if "tools_initialized" not in st.session_state:
    st.session_state.tools_initialized = False
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "last_analyzed_text" not in st.session_state:
    st.session_state.last_analyzed_text = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# API Keys
GROQ_API_KEY = "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"

# LangSmith Configuration
LANGSMITH_API_KEY = "ls__a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"
LANGSMITH_PROJECT = "Toxicity-Agentic-RAG-Application"
LANGSMITH_TRACING_V2 = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

@st.cache_resource
def load_vectorstore():
    """Load the FAISS vector store with better error handling"""
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
        except ImportError as ie:
            st.error(f"Import error: {ie}")
            st.info("Please install required packages: pip install langchain-community sentence-transformers")
            return None
        
        embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = FAISS.load_local(
            "faiss_vector_store",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.info("Make sure the faiss_vector_store folder exists and contains the index files.")
        return None

@st.cache_resource
def initialize_chat_model():
    """Initialize the chat model"""
    try:
        chat_model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            groq_api_key=GROQ_API_KEY,
            max_tokens=1024
        )
        return chat_model
    except Exception as e:
        st.error(f"Error initializing chat model: {e}")
        return None


def safe_invoke(model, prompt):
    """Safe model invocation with error handling"""
    try:
        response = model.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Error: {str(e)}"

def detect_user_intent(user_input: str) -> Dict[str, Any]:
    """
    LLM-powered intent detection to route queries to the right tool
    Returns: {"intent": str, "confidence": float, "reasoning": str}
    """
    # Get the chat model from session state
    chat_model = st.session_state.get("chat_model")
    
    # Build context about previous analysis
    has_previous_analysis = bool(st.session_state.last_analysis)
    previous_text = st.session_state.last_analyzed_text if has_previous_analysis else "None"
    
    # LLM-based intent detection
    intent_prompt = f"""You are an intent classification expert for a toxicity detection system. Analyze the user's query and classify it into ONE of these categories:

                    1. **followup** - Questions about a PREVIOUS toxicity analysis or requests for MORE/ADDITIONAL suggestions/paraphrases/alternatives
                    Examples: 
                    - "What was the severity?"
                    - "Can you suggest more alternatives?"
                    - "Give me other ways to say this"
                    - "What were the categories?"
                    - "More paraphrased versions"
                    - "Any other suggestions?"

                    2. **pattern** - Requests to FIND patterns, examples, or search the database
                    Examples:
                    - "Show me examples of insults"
                    - "Find threat patterns"
                    - "What are common toxic phrases?"
                    - "Give me examples of toxic comments"

                    3. **educational** - Questions asking WHY something is harmful or seeking to UNDERSTAND toxicity concepts
                    Examples:
                    - "Why is this harmful?"
                    - "What makes hate speech dangerous?"
                    - "Explain the impact of insults"

                    4. **analyze** - NEW text to analyze for toxicity (default for direct text)
            Examples:
            - "You are stupid"
            - "Is 'idiot' toxic?"
            - "Analyze: I hate you"

            CONTEXT:
            - Previous analysis exists: {has_previous_analysis}
            - Previously analyzed text: "{previous_text}"

            USER QUERY: "{user_input}"

            CRITICAL RULES:
            - If user asks for "more", "other", "additional", "another", "alternatives" → FOLLOWUP
            - If user asks "what was", "what were", "the severity", "the categories" → FOLLOWUP
            - If requesting database search/patterns/examples → PATTERN
            - If asking "why" or educational explanation → EDUCATIONAL
            - If providing new text to analyze → ANALYZE
            - When in doubt between FOLLOWUP and ANALYZE, choose FOLLOWUP if previous analysis exists

            Respond in EXACTLY this JSON format (no markdown, no extra text):
            {{"intent": "followup|pattern|educational|analyze", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        response = safe_invoke(chat_model, intent_prompt)
        
        # Parse JSON response
        # Extract JSON from response (handle cases where model adds extra text)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Override: if no previous analysis exists and intent is followup, change to analyze
            if result["intent"] == "followup" and not has_previous_analysis:
                result = {
                    "intent": "analyze",
                    "confidence": 0.8,
                    "reasoning": "No previous analysis, treating as new analysis"
                }
            
            return result
        else:
            # Failed to parse, fallback
            return {
                "intent": "analyze",
                "confidence": 0.5,
                "reasoning": "Failed to parse LLM response, defaulting to analyze"
            }
            
    except Exception as e:
        logging.error(f"Intent detection error: {e}")
        # Fallback: default to analyze
    return {
        "intent": "analyze",
            "confidence": 0.5,
            "reasoning": f"Error in LLM intent detection: {str(e)}"
        }

def create_tools(vectorstore, chat_model):
    """Create enhanced tools with dedicated Q&A tool for follow-ups"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def toxicity_analyzer(query: str) -> str:
        """Analyze toxicity of NEW text"""
        try:
            docs = retriever.get_relevant_documents(f"toxicity analysis: {query}")
            context = "\n".join([doc.page_content for doc in docs[:2]])
            
            prompt = f"""You are a toxicity analysis expert. Analyze this text: "{query}"

            Context from database: {context}

            CLASSIFICATION RULES:
            - Genuine compliments (beautiful, handsome, nice, good job) = NOT TOXIC
            - Insults, threats, harassment, hate speech = TOXIC
            - Use positive intent assumption for borderline cases

            AVAILABLE CATEGORIES (use if toxic):
            - toxic: General toxic content
            - severe_toxic: Extremely harmful content  
            - obscene: Profanity and vulgar language
            - threat: Threatening language
            - insult: Direct insults and name-calling
            - identity_hate: Attacks based on identity (race, gender, etc.)

            YOU MUST RESPOND IN EXACTLY THIS FORMAT WITH EACH FIELD ON A NEW LINE:

            TOXICITY: [Yes/No]

            SEVERITY: [1-10 if toxic, N/A if not toxic]

            CATEGORIES: [List applicable categories from above, or N/A if not toxic]

            EXPLANATION: [Clear reasoning based on content and intent]

            PARAPHRASED: [If toxic, rewrite non-toxically; if not toxic, write "Not needed"]

            SUGGESTION: [If toxic, provide communication advice; if not toxic, write "None needed"]

            ANALYSIS SUMMARY: [Brief assessment focusing on intent and impact]

            IMPORTANT: Put a blank line between each section. Provide ONLY the structured analysis above, nothing else."""
            
            response = safe_invoke(chat_model, prompt)
            
            # Clean up the response first (remove extra spaces, normalize)
            response = response.strip()
            
            # Format output with proper line breaks for better UI display
            # Use regex to ensure proper spacing regardless of input format
            formatted_response = re.sub(r'\bTOXICITY:', '\n**TOXICITY:**', response)
            formatted_response = re.sub(r'\bSEVERITY:', '\n\n**SEVERITY:**', formatted_response)
            formatted_response = re.sub(r'\bCATEGORIES:', '\n\n**CATEGORIES:**', formatted_response)
            formatted_response = re.sub(r'\bEXPLANATION:', '\n\n**EXPLANATION:**', formatted_response)
            formatted_response = re.sub(r'\bPARAPHRASED:', '\n\n**PARAPHRASED:**', formatted_response)
            formatted_response = re.sub(r'\bSUGGESTION:', '\n\n**SUGGESTION:**', formatted_response)
            formatted_response = re.sub(r'\bANALYSIS SUMMARY:', '\n\n**ANALYSIS SUMMARY:**', formatted_response)
            
            # Clean up any multiple consecutive newlines (more than 2)
            formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
            
            # Store analysis for follow-up questions (store original, not formatted)
            st.session_state.last_analysis = response
            st.session_state.last_analyzed_text = query
            st.session_state.analysis_history.append({
                "text": query,
                "analysis": response,
                "timestamp": time.time(),
                "tool": "ToxicityAnalyzer"
            })
            
            return formatted_response
            
        except Exception as e:
            return f"Analysis Error: {e}"

    def pattern_finder(query: str) -> str:
        """Find toxic patterns and examples"""
        try:
            # Handle general example requests
            query_lower = query.lower()
            if 'general' in query_lower or 'some examples' in query_lower or 'give me' in query_lower:
                search_query = "toxic comments examples insults threats"
            else:
                search_query = f"toxic patterns: {query}"
            
            docs = retriever.get_relevant_documents(search_query)
            context = "\n".join([doc.page_content for doc in docs[:4]])
            
            prompt = f"""You are a toxicity pattern analysis expert. The user asked: "{query}"

            Context from database (real toxic examples):
            {context}

            YOU MUST RESPOND IN EXACTLY THIS FORMAT WITH EACH FIELD ON A NEW LINE:

            COMMON PATTERNS:
            - [Pattern 1 with severity level]
            - [Pattern 2 with severity level] 
            - [Pattern 3 with severity level]

            PATTERN ANALYSIS:
            - Why these patterns are harmful
            - Common characteristics
            - Detection indicators

            EXAMPLES (from database):
            - [Real toxic example 1]
            - [Real toxic example 2]
            - [Real toxic example 3]

            IMPORTANT: Put a blank line between each section. Provide ONLY the structured analysis above, nothing else. Use REAL examples from the context provided."""
            
            response = safe_invoke(chat_model, prompt)
            
            # Clean up the response first (remove extra spaces, normalize)
            response = response.strip()
            
            # Format output with proper line breaks for better UI display
            # Use regex to ensure proper spacing regardless of input format
            formatted_response = re.sub(r'\bCOMMON PATTERNS:', '\n**COMMON PATTERNS:**', response)
            formatted_response = re.sub(r'\bPATTERN ANALYSIS:', '\n\n**PATTERN ANALYSIS:**', formatted_response)
            formatted_response = re.sub(r'\bEXAMPLES \(from database\):', '\n\n**EXAMPLES (from database):**', formatted_response)
            formatted_response = re.sub(r'\bEXAMPLES:', '\n\n**EXAMPLES:**', formatted_response)
            
            # Clean up any multiple consecutive newlines (more than 2)
            formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
            
            # Store in history but DON'T overwrite last_analysis (preserve toxicity analysis for follow-ups)
            st.session_state.analysis_history.append({
                "text": query,
                "analysis": response,
                "timestamp": time.time(),
                "tool": "PatternFinder"
            })
            
            return formatted_response
            
        except Exception as e:
            return f"Pattern Search Error: {e}"

    def educational_guide(query: str) -> str:
        """Provide educational guidance"""
        try:
            docs = retriever.get_relevant_documents(f"education harm: {query}")
            base_context = "\n".join([doc.page_content for doc in docs[:2]])
            
            prompt = f"""You are a toxicity education expert. Explain why this is harmful: "{query}"

            Educational context: {base_context}

            YOU MUST RESPOND IN EXACTLY THIS FORMAT WITH EACH FIELD ON A NEW LINE:

            HARM TYPE: [specific toxicity category - insult, threat, hate speech, etc.]

            IMPACT: [who gets hurt and how - psychological, emotional, social effects]

            WHY HARMFUL: [detailed explanation of psychological/social effects]

            BETTER APPROACH: [constructive alternatives and suggestions]

            EDUCATIONAL SUMMARY: [key learning points about this type of harm]

            IMPORTANT: Put a blank line between each section. Provide ONLY the structured education above, nothing else."""
            
            response = safe_invoke(chat_model, prompt)
            
            # Clean up the response first (remove extra spaces, normalize)
            response = response.strip()
            
            # Format output with proper line breaks for better UI display
            # Use regex to ensure proper spacing regardless of input format
            formatted_response = re.sub(r'\bHARM TYPE:', '\n**HARM TYPE:**', response)
            formatted_response = re.sub(r'\bIMPACT:', '\n\n**IMPACT:**', formatted_response)
            formatted_response = re.sub(r'\bWHY HARMFUL:', '\n\n**WHY HARMFUL:**', formatted_response)
            formatted_response = re.sub(r'\bBETTER APPROACH:', '\n\n**BETTER APPROACH:**', formatted_response)
            formatted_response = re.sub(r'\bEDUCATIONAL SUMMARY:', '\n\n**EDUCATIONAL SUMMARY:**', formatted_response)
            
            # Clean up any multiple consecutive newlines (more than 2)
            formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
            
            # Store in history but DON'T overwrite last_analysis (preserve toxicity analysis for follow-ups)
            st.session_state.analysis_history.append({
                "text": query,
                "analysis": response,
                "timestamp": time.time(),
                "tool": "EducationalGuide"
            })
            
            return formatted_response
            
        except Exception as e:
            return f"Education Error: {e}"

    
    def conversation_assistant(query: str) -> str:
        """Answer follow-up questions about previous toxicity analysis"""
        try:
            # Find the most recent TOXICITY ANALYSIS from history
            last_toxicity_analysis = None
            last_toxic_text = None
            
            # Search backwards through history for the last ToxicityAnalyzer result
            for item in reversed(st.session_state.analysis_history):
                if item.get("tool") == "ToxicityAnalyzer":
                    last_toxicity_analysis = item.get("analysis")
                    last_toxic_text = item.get("text")
                    break
            
            # Fallback to session state if no history found
            if not last_toxicity_analysis:
                if st.session_state.last_analysis:
                    last_toxicity_analysis = st.session_state.last_analysis
                    last_toxic_text = st.session_state.last_analyzed_text
                else:
                    return "❌ No toxicity analysis found. Please analyze some text first, then ask follow-up questions."
            
            # Classify the follow-up type with THREE categories
            followup_type_prompt = f"""You are classifying a follow-up question about a toxicity analysis.

                ORIGINAL ANALYZED TEXT: "{last_toxic_text}"

                USER'S FOLLOW-UP QUESTION: "{query}"

                Classify this follow-up question into ONE of these types:

                1. **EXTRACTION** - User wants to retrieve information FROM the existing analysis
                Examples:
                - "what was the severity?"
                - "what were the categories?"
                - "tell me the explanation"
                - "show me the paraphrased version"
                - "what was in the suggestion?"

                2. **GENERATION** - User wants COMPLETELY NEW content (not based on existing alternatives)
                Examples:
                - "give me MORE suggestions"
                - "what are OTHER alternatives?"
                - "can you provide ADDITIONAL paraphrases?"
                - "suggest some OTHER ways"
                - "give me 5 more examples"
                
                3. **REFINEMENT** - User wants to MODIFY or REFINE existing content from the analysis
                Examples:
                - "make it more casual"
                - "shorter version"
                - "less formal"
                - "more professional"
                - "simplify this"
                - "make it friendlier"
                - "can you rephrase that?"
                - "make it more direct"

                Respond with ONLY ONE WORD: either "EXTRACTION" or "GENERATION" or "REFINEMENT"

                Your classification:"""
            
            followup_type = safe_invoke(chat_model, followup_type_prompt).strip().upper()
            
            # Handle EXTRACTION requests
            if "EXTRACTION" in followup_type:
                prompt = f"""You are extracting information from a previous toxicity analysis.

            ORIGINAL TEXT ANALYZED: "{last_toxic_text}"

            FULL ANALYSIS:
            {last_toxicity_analysis}

            USER'S QUESTION: "{query}"

            INSTRUCTIONS:
            - If asking for SEVERITY → extract the SEVERITY value
            - If asking for CATEGORIES → extract the CATEGORIES list
            - If asking for EXPLANATION → extract the EXPLANATION section
            - If asking for PARAPHRASE → extract the PARAPHRASED version
            - If asking for SUGGESTION → extract the SUGGESTION section
            - If asking for SUMMARY → extract the ANALYSIS SUMMARY

            Just extract and return the relevant information from the analysis above. Be concise and direct."""
                
                response = safe_invoke(chat_model, prompt)
                return f"💬 **Follow-up Answer:**\n\n{response}\n\n_From analysis of: \"{last_toxic_text[:50]}...\"_"
            
            # Handle REFINEMENT requests
            elif "REFINEMENT" in followup_type:
                prompt = f"""You are refining content from a previous toxicity analysis based on user feedback.

                    ORIGINAL TOXIC TEXT: "{last_toxic_text}"

                PREVIOUS ANALYSIS (contains the content to refine):
                {last_toxicity_analysis}

                USER'S REFINEMENT REQUEST: "{query}"

                YOUR TASK:
                1. Identify what content from the analysis the user wants refined (paraphrases, suggestions, explanations)
                2. Apply the requested modification style:
                - "more casual" → use conversational language, contractions, friendly tone
                - "shorter" → condense while keeping key points
                - "less formal" → remove jargon, use simpler words
                - "more professional" → use formal language, complete sentences
                - "friendlier" → add warmth, empathy, positive framing
                - "more direct" → remove fluff, get straight to the point
                - "simpler" → use basic vocabulary, shorter sentences

                3. Keep the SAME underlying message/intent, just adjust the style

                IMPORTANT:
                - Maintain the non-toxic nature of the content
                - Don't change the core meaning
                - Apply the style modification consistently
                - If refining alternatives/paraphrases, provide 3-5 refined versions

                Your refined content:"""
                
                response = safe_invoke(chat_model, prompt)
                return f"✨ **Refined Version:**\n\n{response}\n\n_Refined from analysis of: \"{last_toxic_text[:50]}...\"_"
            
            # Handle GENERATION requests
            else:  # GENERATION
                # Get relevant examples from database for generating new content
                docs = retriever.get_relevant_documents(f"paraphrase alternatives suggestions: {last_toxic_text}")
                context_examples = "\n".join([doc.page_content for doc in docs[:3]])
                
                # First, extract what was ALREADY provided to explicitly exclude it
                extraction_prompt = f"""Extract ONLY the suggestions/alternatives/paraphrases that were already provided in this analysis.

                    ANALYSIS:
                    {last_toxicity_analysis}

                    List them in a simple numbered format. Extract ONLY what was already given, nothing more."""
                
                already_provided = safe_invoke(chat_model, extraction_prompt)
                
                # Now generate NEW content with explicit exclusion
                prompt = f"""You are generating ADDITIONAL content for a toxicity analysis.

                ORIGINAL TOXIC TEXT: "{last_toxic_text}"

                CONTEXT: The user previously received some suggestions/alternatives and now wants MORE.

                ❌ CONTENT ALREADY PROVIDED (DO NOT REPEAT THESE):
                {already_provided}

                ✅ RELEVANT EXAMPLES FROM DATABASE (use these as inspiration):
                {context_examples}

                USER'S REQUEST: "{query}"

                YOUR TASK - Generate COMPLETELY NEW content:

                If asking for MORE SUGGESTIONS:
                - Provide 3-5 DIFFERENT communication strategies NOT mentioned above
                - Think about: conflict resolution techniques, emotional regulation, perspective-taking
                - Examples: active listening, using "I" statements, taking breaks, seeking common ground
                - Be specific and actionable

                If asking for MORE ALTERNATIVES/PARAPHRASES:
                - Create 3-5 DIFFERENT ways to express the same intent
                - Use DIFFERENT vocabulary and sentence structures than what's listed above
                - Think about different tones: assertive, curious, collaborative, reflective
                - Draw inspiration from the database examples
                - Make them conversational and natural

                If asking for OTHER WAYS:
                - Analyze the underlying emotion (anger, frustration, hurt, disagreement)
                - Generate NEW approaches not mentioned above
                - Consider: reframing perspective, addressing needs, expressing vulnerability
                - Number them clearly (1., 2., 3., etc.)

                🔥 CRITICAL RULES:
                1. DO NOT USE any phrases, words, or ideas from "CONTENT ALREADY PROVIDED"
                2. Check your output - if ANY part resembles what's above, REPLACE IT
                3. Generate from the DATABASE EXAMPLES and your own knowledge
                4. Aim for variety in structure, tone, and approach
                5. If generating paraphrases, ensure each uses a DISTINCT communication style

                Your BRAND NEW content (verify it's different from what was already provided):"""
                
                response = safe_invoke(chat_model, prompt)
                return f"💡 **Additional Content:**\n\n{response}\n\n_Based on analysis of: \"{last_toxic_text[:50]}...\"_"
            
        except Exception as e:
            return f"Conversation Error: {e}"
    
    tools = {
        "analyze": toxicity_analyzer,
        "pattern": pattern_finder,
        "educational": educational_guide,
        "followup": conversation_assistant
    }
    
    return tools

def setup_runtime():
    """Single setup to initialize vectorstore, chat model, memory, tools, RAG chain, and ReAct agent."""
    status = {
        "vectorstore_loaded": False,
        "chat_initialized": False
    }

    # Vectorstore
    vectorstore = load_vectorstore()
    if not vectorstore:
        st.error("❌ Failed to load vectorstore. Cannot proceed without it.")
        return status
    
    st.session_state.vectorstore = vectorstore
    status["vectorstore_loaded"] = True

    # Chat model
    chat_model = initialize_chat_model()
    if not chat_model:
        return status
    st.session_state.chat_model = chat_model
    status["chat_initialized"] = True

    # Memory
    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        return_messages=True
    )

    # Callable tools
    tools = create_tools(vectorstore, chat_model)

    # ReAct agent tools with detailed descriptions
    agent_tools = [
        Tool(
            name="toxicity_analyzer",
            func=tools["analyze"],
            description="Use this to analyze text for toxicity, insults, threats, or harmful content. Input should be the exact text to analyze. Always use this for phrases like 'You are stupid', 'I hate you', etc."
        ),
        Tool(
            name="pattern_finder",
            func=tools["pattern"],
            description="Use this to find patterns and examples of toxic behavior. Input should describe the pattern you want to find, like 'insult patterns' or 'threat examples'."
        ),
        Tool(
            name="educational_guide",
            func=tools["educational"],
            description="Use this to explain why certain language is harmful and suggest better alternatives. Input should be the concept or phrase to explain."
        ),
        Tool(
            name="followup_qa",
            func=tools["followup"],
            description="Use this ONLY to answer follow-up questions about a previous toxicity analysis (severity, categories, paraphrasing, etc.). Do not use for new analysis."
        )
    ]

    # ReAct agent with manual prompt template optimized for direct tool output
    react_prompt = PromptTemplate.from_template("""You are a toxicity detection assistant with access to specialized tools. You have access to the following tools:

                    {tools}

                    IMPORTANT INSTRUCTIONS:
                    - If the user provides text to analyze for toxicity, use toxicity_analyzer and return its EXACT output without modification
                    - If the user asks for patterns, use pattern_finder and return its EXACT output
                    - If the user asks why something is harmful, use educational_guide and return its EXACT output
                    - If the user asks about previous analysis, use followup_qa and return its EXACT output
                    - DO NOT summarize or paraphrase tool outputs - return them as-is

                    Use the following format:

                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question

                    Begin!

                    Question: {input}
                    Thought:{agent_scratchpad}""")
    
    agent_graph = create_react_agent(chat_model, agent_tools, react_prompt)
    agent = AgentExecutor(
        agent=agent_graph,
        tools=agent_tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate"
    )

    # Save in session
    st.session_state.memory = memory
    st.session_state.tools = tools
    st.session_state.agent_tools = agent_tools
    st.session_state.agent = agent
    st.session_state.tools_initialized = True

    return status


def process_user_input(user_input: str, tools, memory):
    """
    Fast direct tool routing based on intent detection.
    Falls back to agent only if needed.
    """
    
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        # Detect user intent for fast routing
        intent_result = detect_user_intent(user_input)
        intent = intent_result["intent"]
        
        # Direct tool call (fast path)
        tool_func = tools.get(intent)
        
        if tool_func:
            # Show what we're doing
            intent_labels = {
                "followup": "💬 Answering follow-up question",
                "analyze": "🔍 Analyzing toxicity",
                "pattern": "📊 Finding patterns",
                "educational": "📚 Providing education"
            }
            
            with st.spinner(f"{intent_labels.get(intent, 'Processing')}..."):
                result = tool_func(user_input)
            
            # Add to memory
            if memory and hasattr(memory, 'chat_memory'):
                memory.chat_memory.add_user_message(user_input)
                memory.chat_memory.add_ai_message(result)
            
            return result
        
        # Fallback to agent for complex queries
        agent = st.session_state.get("agent")
        if agent is not None:
            with st.spinner("🤖 Agent analyzing query..."):
                agent_response = agent.invoke({"input": user_input})
                
                if isinstance(agent_response, dict) and agent_response.get("output"):
                    result = agent_response["output"]
                else:
                    result = str(agent_response)
                
                if memory and hasattr(memory, 'chat_memory'):
                    memory.chat_memory.add_user_message(user_input)
                    memory.chat_memory.add_ai_message(result)
                
                return result
        else:
            return "❌ System not initialized. Please click 'Initialize System' in the sidebar."
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.error(error_msg)
        return error_msg

def main():
    st.title("🛡️ Toxicity Detection & Content Moderation")
    st.markdown("**Intelligent AI system with smart routing and follow-up support**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Load vector store, chat model, tools, memory, and agent
        if st.button("🔄 Initialize System"):
            with st.spinner("Loading system components..."):
                status = setup_runtime()
                
                if not status.get("vectorstore_loaded"):
                    st.error("❌ Failed to load vector store - system cannot initialize.")
                    return

                if not status.get("chat_initialized"):
                    st.error("❌ Failed to initialize chat model")
                    return

                st.success("✅ Vector store loaded!")
                st.success("✅ System ready! Smart routing enabled.")
        
        st.markdown("---")
        
        # Context status display
        st.markdown("### 💭 Analysis Context")
        if st.session_state.last_analyzed_text:
            st.write(f"**Last Analyzed:** '{st.session_state.last_analyzed_text[:60]}...'")
            st.write(f"**Total Analyses:** {len(st.session_state.analysis_history)}")
            st.success("✅ Follow-up questions enabled")
        else:
            st.write("**Status:** No analysis yet")
            st.info("ℹ️ Analyze text first to enable follow-ups")
        
        # Show recent analyses
        if len(st.session_state.analysis_history) > 0:
            with st.expander("📊 Recent Analyses"):
                for i, item in enumerate(reversed(st.session_state.analysis_history[-3:]), 1):
                    st.write(f"**{i}. [{item['tool']}]**")
                    st.write(f"Text: {item['text'][:50]}...")
                    st.write(f"Time: {time.strftime('%H:%M:%S', time.localtime(item['timestamp']))}")
                    st.markdown("---")
        
        st.markdown("---")
        
        # Intent routing info
        st.markdown("### 🎯 Smart Routing")
        st.markdown("""
        System automatically detects intent:
        
        **🔍 Toxicity Analysis**  
        "You are stupid" → Analysis
        
        **💬 Follow-up Questions**  
        "What was the severity?" → Q&A
        
        **📊 Pattern Search**  
        "Show me insult patterns" → Patterns
        
        **📚 Educational**  
        "Why is X harmful?" → Education
        """)
        
        st.markdown("---")
        
        # LangSmith Tracing
        st.markdown("### 📊 LangSmith Tracing")
        if st.button("🔗 Open LangSmith Dashboard"):
            st.markdown(f"[🚀 View Traces](https://smith.langchain.com/projects/{LANGSMITH_PROJECT})")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All"):
            st.session_state.messages = []
            st.session_state.last_analysis = None
            st.session_state.last_analyzed_text = None
            st.session_state.analysis_history = []
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.success("✅ Cleared!")
            st.rerun()
    
    # Main chat interface
    if not st.session_state.tools_initialized:
        st.warning("⚠️ Please initialize the system from the sidebar first!")
        st.markdown("""
        ## 🛡️ AI-Powered Toxicity Detection with Smart Routing
        
        **No more agent iteration errors! Direct tool routing for instant results.**
        
        ### ✨ Features:
        
        🎯 **Smart Intent Detection**  
        Automatically routes your query to the right tool
        
        💬 **Natural Follow-ups**  
        Ask questions about your previous analyses
        
        📊 **Context Tracking**  
        System remembers your analysis history
        
        ### How to Use:
        
        1. **Initialize** - Click "🔄 Initialize System" 
        2. **Analyze** - Type text to analyze: "You are stupid"
        3. **Follow-up** - Ask questions: "What was the severity?"
        4. **Explore** - Request patterns: "Show me threat patterns"
        5. **Learn** - Ask: "Why is hate speech harmful?"
        
        ### Example Conversation:
        
        **You:** "You are an idiot"  
        **System:** [Analyzes toxicity with severity, categories, paraphrase]
        
        **You:** "What was the severity level?"  
        **System:** [Extracts severity from previous analysis]
        
        **You:** "Show me similar insult patterns"  
        **System:** [Finds related toxic patterns]
        
        ---
        
        Click "🔄 Initialize System" to get started!
        """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Analyze text, ask questions, or request patterns..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Processing..."):
                response = process_user_input(
                    prompt, 
                    st.session_state.tools,
                    st.session_state.memory
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()