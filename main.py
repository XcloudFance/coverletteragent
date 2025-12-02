"""
Deep Academic Cold Email Generator with Gradio Interface
Features:
- Multi-professor parallel email generation
- University professor search
- Clean web interface with individual email viewer
- Async processing

Usage:
1. Install: pip install gradio crewai langchain-community duckduckgo-search arxiv PyPDF2 requests
2. Update OpenAI API key below
3. python academic_email_gradio.py
"""

import os
import io
import requests
import arxiv
import asyncio
import concurrent.futures
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import gradio as gr

# ===== Environment Setup =====
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_MODEL_NAME"] = ""

# ===== Tools =====

@tool("DuckDuckGo Search")
def duckduckgo_search_tool(question: str) -> str:
    """Use DuckDuckGo Search to find general information."""
    search = DuckDuckGoSearchRun()
    return search.run(question)

@tool("Arxiv Paper Search")
def search_arxiv_papers(professor_name: str) -> str:
    """Searches ArXiv for the most recent papers by the professor."""
    try:
        search = arxiv.Search(
            query=f'au:"{professor_name}"',
            max_results=1,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        result_str = ""
        for result in search.results():
            result_str += f"Title: {result.title}\n"
            result_str += f"Published: {result.published}\n"
            result_str += f"Abstract: {result.summary}\n"
            result_str += f"PDF_URL: {result.pdf_url}\n"
            return result_str
            
        return "No ArXiv papers found."
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

@tool("Online PDF Reader")
def read_online_pdf(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            with io.BytesIO(response.content) as f:
                reader = PdfReader(f)
                text = ""
                pages_to_read = [0, 1] 
                if len(reader.pages) > 2:
                    pages_to_read.append(len(reader.pages) - 1)
                
                for page_num in pages_to_read:
                    if page_num < len(reader.pages):
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += reader.pages[page_num].extract_text()
                
                return text[:10000]
        else:
            return f"Failed to download PDF. Status: {response.status_code}"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# ===== Email Template Guidelines =====

EMAIL_GUIDELINES = """
Writing Principles:
- **80/20 Rule**: 80% about professor's work, 20% about applicant
- **Genuine Understanding**: Show you read their specific papers
- **No Excessive Praise**: Be sincere, not flattering. Say what inspired you, but keep it grounded.
- **Refer to Resume**: DON'T list your achievements - mention resume instead
- **Concise**: Under 250 words
- **Professional & Warm**: Respectful but not overly formal

Example Opening (Good):
"Your recent work on [specific method] addresses [problem] in an interesting way. 
The approach to [specific technique] made me reconsider how I've been thinking about [related area]."

Example Opening (Too Flattering - AVOID):
"Your groundbreaking, revolutionary work has completely transformed the field and I am in awe..."

Be specific, be genuine, be humble.
"""

# ===== Single Professor Email Generator =====

class SingleProfessorEmailGenerator:
    """Generates email for ONE professor only - independent thread"""
    
    def __init__(self):
        self.crew = None
    
    def _create_agents(self):
        """Create agents for this specific professor"""
        
        professor_researcher = Agent(
            role='Professor Background Researcher',
            goal='Find professor homepage, lab, and general research interests',
            backstory="""You find academic homepages and research group information.
            Focus on current position, lab name, and broad research areas.""",
            tools=[duckduckgo_search_tool],
            verbose=False,
            allow_delegation=False
        )
        
        paper_analyst = Agent(
            role='Paper Deep Dive Analyst',
            goal='Find latest ArXiv paper, read PDF, extract technical details',
            backstory="""You read actual PDFs to find specific methodology, model names,
            and technical insights. You go beyond the abstract.""",
            tools=[search_arxiv_papers, read_online_pdf],
            verbose=False,
            allow_delegation=False
        )
        
        alignment_analyst = Agent(
            role='Alignment Analyst',
            goal='Find genuine connections between applicant and professor research',
            backstory="""You identify real alignment points without forcing connections.
            You focus on understanding and capability match, not listing achievements.""",
            verbose=False,
            allow_delegation=False
        )
        
        email_writer = Agent(
            role='Academic Email Writer',
            goal='Write sincere, well-researched emails that show genuine interest',
            backstory=f"""You write academic emails that get responses.
            You are SINCERE and GROUNDED - you avoid excessive praise or flattering language.
            You show understanding and interest, not adoration.
            
            {EMAIL_GUIDELINES}""",
            verbose=False,
            allow_delegation=False
        )
        
        return professor_researcher, paper_analyst, alignment_analyst, email_writer
    
    def _create_tasks(self, professor_name: str, university: str, 
                     applicant_background: str, specific_ask: str,
                     agents: tuple):
        """Create tasks for this professor"""
        
        prof_researcher, paper_analyst, alignment_analyst, email_writer = agents
        
        research_task = Task(
            description=f"""Research {professor_name} at {university}.
            Find: homepage, research group, main areas.
            Search: "{professor_name} {university} homepage"
            Context: {applicant_background[:200]}...""",
            expected_output="Profile with position, research areas, lab info",
            agent=prof_researcher
        )
        
        paper_task = Task(
            description=f"""
            1. Find latest ArXiv paper by {professor_name}
            2. Get PDF URL and read it
            3. Extract: title, problem, methodology/model name, key insight""",
            expected_output="Technical analysis with title, methodology, insights",
            agent=paper_analyst
        )
        
        alignment_task = Task(
            description=f"""Analyze alignment between applicant and professor's work.
            
            Applicant: {applicant_background}
            
            Focus on:
            1. Connection to SPECIFIC paper (from Paper Analysis)
            2. How applicant's background helps them understand this work
            3. One-sentence summary of applicant capability
            4. Natural talking points
            
            Remember: 80% professor, 20% applicant.""",
            expected_output="Alignment analysis with specific connections",
            agent=alignment_analyst,
            context=[research_task, paper_task]
        )
        
        writing_task = Task(
            description=f"""Write cold email to professor.
            
            Requirements:
            1. Specific subject line mentioning paper topic
            2. Opening shows you know their SPECIFIC recent work
            3. 80% discussing their work and your understanding
            4. 20% about yourself (brief, refer to resume)
            5. NO detailed listing of your achievements
            6. Ask: {specific_ask}
            7. CRITICAL: Be SINCERE, not flattering. Show genuine interest and understanding,
               but avoid excessive praise. Say what inspired you specifically, not generic compliments.
            
            Length: Under 250 words
            Tone: Professional, humble, genuinely interested
            
            {EMAIL_GUIDELINES}""",
            expected_output="Complete email with subject and body (250 words)",
            agent=email_writer,
            context=[alignment_task]
        )
        
        return [research_task, paper_task, alignment_task, writing_task]
    
    def generate(self, professor_name: str, university: str, 
                applicant_background: str, specific_ask: str) -> Dict:
        """Generate email for ONE professor"""
        
        try:
            agents = self._create_agents()
            tasks = self._create_tasks(professor_name, university, 
                                      applicant_background, specific_ask, agents)
            
            self.crew = Crew(
                agents=list(agents),
                tasks=tasks,
                process=Process.sequential,
                verbose=False
            )
            
            result = self.crew.kickoff()
            
            return {
                "professor": professor_name,
                "university": university,
                "status": "success",
                "email": str(result)
            }
            
        except Exception as e:
            return {
                "professor": professor_name,
                "university": university,
                "status": "error",
                "email": f"Error generating email: {str(e)}"
            }

# ===== University Professor Finder =====

class UniversityProfessorFinder:
    """Finds professors at a university - separate independent system"""
    
    def __init__(self):
        self.crew = None
    
    def find_professors(self, university_name: str, department: str = "Computer Science") -> str:
        """Search for professors at a university"""
        
        # Create specialized agents for university search
        web_researcher = Agent(
            role='University Website Researcher',
            goal=f'Find faculty list for {department} at {university_name}',
            backstory="""You excel at finding faculty directories on university websites.
            You search department pages, faculty listings, and research group pages.""",
            tools=[duckduckgo_search_tool],
            verbose=False,
            allow_delegation=False
        )
        
        list_compiler = Agent(
            role='Professor List Compiler',
            goal='Compile a clean list of professors with their research areas',
            backstory="""You organize faculty information into clean lists.
            You extract names, titles, and research areas from web content.""",
            verbose=False,
            allow_delegation=False
        )
        
        # Tasks
        search_task = Task(
            description=f"""Find the faculty/professor list for {department} at {university_name}.
            
            Search strategies:
            - "{university_name} {department} faculty"
            - "{university_name} {department} professors"
            - "{university_name} CS department people"
            
            Look for: official department pages, faculty directories, research groups.
            Extract: Professor names, titles, and research areas if available.""",
            expected_output="Raw information about faculty members",
            agent=web_researcher
        )
        
        compile_task = Task(
            description="""Compile the faculty information into a clean list.
            
            Format each professor as:
            - Name: [Full Name]
            - Title: [Professor/Associate Professor/Assistant Professor]
            - Research: [Brief research areas]
            
            Sort by rank (Full Prof -> Associate -> Assistant).
            Only include tenure-track faculty.""",
            expected_output="Formatted list of professors",
            agent=list_compiler,
            context=[search_task]
        )
        
        self.crew = Crew(
            agents=[web_researcher, list_compiler],
            tasks=[search_task, compile_task],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            result = self.crew.kickoff()
            return str(result)
        except Exception as e:
            return f"Error finding professors: {str(e)}"

# ===== Multi-Professor Parallel Generator =====

def generate_multiple_emails(professors_data: List[Dict], applicant_background: str, 
                            specific_ask: str, max_concurrent: int = 3, 
                            progress=gr.Progress()) -> Tuple[Dict[str, str], List[str]]:
    """
    Generate emails for multiple professors in parallel
    Returns: (dict of emails, list of professor names for dropdown)
    """
    
    results = []
    total = len(professors_data)
    
    def generate_single(data, idx):
        progress((idx + 1) / total, desc=f"Processing {data['name']} ({idx+1}/{total})...")
        generator = SingleProfessorEmailGenerator()
        return generator.generate(
            professor_name=data['name'],
            university=data['university'],
            applicant_background=applicant_background,
            specific_ask=specific_ask
        )
    
    # Use ThreadPoolExecutor for parallel processing
    # Automatically queues if professors_data > max_concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(generate_single, data, i) 
                  for i, data in enumerate(professors_data)]
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Sort results by professor name for consistency
    results.sort(key=lambda x: x['professor'])
    
    # Create email dict and professor list
    email_dict = {}
    professor_list = []
    
    for result in results:
        key = f"{result['professor']} ({result['university']})"
        professor_list.append(key)
        email_dict[key] = result['email']
    
    return email_dict, professor_list

# ===== Gradio Interface =====

def create_interface():
    
    # Global state to store generated emails
    email_storage = {}
    
    with gr.Blocks(title="Academic Email Generator") as app:
        
        gr.Markdown("# 🎓 Academic Cold Email Generator")
        gr.Markdown("Generate personalized emails to professors using AI-powered research")
        
        with gr.Tabs():
            
            # Tab 1: Email Generation
            with gr.Tab("📧 Generate Emails"):
                gr.Markdown("### Generate emails for one or multiple professors")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        resume_input = gr.Textbox(
                            label="Your Background / Resume",
                            placeholder="Paste your resume or background here...",
                            lines=10
                        )
                        
                        specific_ask = gr.Textbox(
                            label="What are you asking for?",
                            value="discuss potential PhD or Research Assistant opportunities for Fall 2025",
                            lines=2
                        )
                        
                        max_concurrent = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Max Concurrent Professors",
                            info="Process this many professors at once"
                        )
                        
                        gr.Markdown("### Professor List")
                        gr.Markdown("Enter professors one per line: `Name | University`")
                        
                        professors_input = gr.Textbox(
                            label="Professors (Name | University)",
                            placeholder="Ruohan Gao | University of Maryland\nTianyi Zhou | University of Maryland",
                            lines=8
                        )
                        
                        generate_btn = gr.Button("🚀 Generate All Emails", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Generated Emails")
                        
                        with gr.Row():
                            professor_selector = gr.Dropdown(
                                label="Select Professor to View Email",
                                choices=[],
                                interactive=True,
                                scale=3
                            )
                            
                            copy_btn = gr.Button("📋 Copy", scale=1)
                        
                        email_display = gr.Textbox(
                            label="Email Content",
                            lines=25,
                            max_lines=40,
                            placeholder="Generated email will appear here after selection..."
                        )
                        
                        status_box = gr.Textbox(
                            label="Generation Status",
                            lines=3,
                            max_lines=5,
                            interactive=False
                        )
                
                gr.Markdown("### Example Format")
                gr.Markdown("""
                ```
                Ruohan Gao | University of Maryland
                Tianyi Zhou | University of Maryland
                Yann LeCun | New York University
                ```
                
                **⚡ How it works:** 
                1. Enter your resume and professor list
                2. Click "Generate All Emails" - system processes them in parallel
                3. After completion, select a professor from the dropdown
                4. View and copy their personalized email
                """)
            
            # Tab 2: Professor Finder
            with gr.Tab("🔍 Find Professors"):
                gr.Markdown("### Search for professors at a university")
                
                with gr.Row():
                    with gr.Column():
                        university_input = gr.Textbox(
                            label="University Name",
                            placeholder="e.g., Stanford University",
                            lines=1
                        )
                        
                        department_input = gr.Textbox(
                            label="Department",
                            value="Computer Science",
                            lines=1
                        )
                        
                        search_btn = gr.Button("🔍 Search Professors", variant="primary")
                    
                    with gr.Column():
                        professor_list_output = gr.Textbox(
                            label="Professor List",
                            lines=25,
                            max_lines=40
                        )
                
                gr.Markdown("""
                **Note:** This will search the university website and compile a list of faculty members.
                Results may vary based on website structure.
                """)
        
        # Event handlers
        def parse_professors(text: str) -> List[Dict]:
            """Parse professor input text"""
            professors = []
            for line in text.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        professors.append({
                            'name': parts[0].strip(),
                            'university': parts[1].strip()
                        })
            return professors
        
        def generate_handler(resume, ask, prof_text, max_conc, progress=gr.Progress()):
            """Generate emails and return dropdown choices + status"""
            if not resume or not prof_text:
                return {
                    professor_selector: gr.Dropdown(choices=[]),
                    email_display: "",
                    status_box: "❌ Please provide your resume and professor list."
                }
            
            professors = parse_professors(prof_text)
            if not professors:
                return {
                    professor_selector: gr.Dropdown(choices=[]),
                    email_display: "",
                    status_box: "❌ No professors found. Use format: Name | University"
                }
            
            # Generate emails
            email_dict, prof_list = generate_multiple_emails(
                professors, resume, ask, int(max_conc), progress
            )
            
            # Store in global state
            nonlocal email_storage
            email_storage = email_dict
            
            status_msg = f"✅ Successfully generated {len(prof_list)} emails!\n"
            status_msg += f"📊 Professors: {', '.join([p.split('(')[0].strip() for p in prof_list])}"
            
            return {
                professor_selector: gr.Dropdown(choices=prof_list, value=prof_list[0] if prof_list else None),
                email_display: email_dict.get(prof_list[0], "") if prof_list else "",
                status_box: status_msg
            }
        
        def update_email_display(selected_professor):
            """Update email display when professor is selected"""
            if selected_professor and selected_professor in email_storage:
                return email_storage[selected_professor]
            return "Select a professor to view their email."
        
        def copy_to_clipboard(email_text):
            """Show copy confirmation"""
            if email_text and email_text != "Select a professor to view their email.":
                return "📋 Email copied! (Use Ctrl+C to copy the text above)"
            return "⚠️ No email to copy"
        
        def search_handler(university, department):
            if not university:
                return "Please enter a university name."
            
            finder = UniversityProfessorFinder()
            return finder.find_professors(university, department)
        
        # Connect events
        generate_btn.click(
            fn=generate_handler,
            inputs=[resume_input, specific_ask, professors_input, max_concurrent],
            outputs=[professor_selector, email_display, status_box]
        )
        
        professor_selector.change(
            fn=update_email_display,
            inputs=[professor_selector],
            outputs=[email_display]
        )
        
        copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[email_display],
            outputs=[status_box]
        )
        
        search_btn.click(
            fn=search_handler,
            inputs=[university_input, department_input],
            outputs=professor_list_output
        )
    
    return app

# ===== Main =====

if __name__ == "__main__":
    app = create_interface()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)