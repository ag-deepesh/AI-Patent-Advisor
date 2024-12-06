import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.system("pip -q install gradio")
os.system("pip -q install datasets")
os.system("pip -q install transformers")
os.system("pip -q install langchain")
os.system("pip -q install sentence_transformers") 
os.system("pip -q install langchain-community faiss-cpu")
os.system("pip -q install openai")
os.system("pip -q install langchain_openai")
os.system("pip -q install tqdm")
os.system("pip -q install pandas")

from src.patent_analyzer import PatentAnalyzer
import query_examples as qe
import gradio as gr

os.environ["HUGGINGFACEHUB_API_TOKEN"] = 
os.environ["HF_TOKEN"] = 
os.environ["OPENAI_API_KEY"] = 

def main():
    analyzer = PatentAnalyzer()
    #Example Usages (Replace with Gradio integration)
    #print(analyzer.prior_art_search("Methods for detecting diseases"))

    
    
    with gr.Blocks() as demo:
        gr.Markdown("# AI Patent Advisor Multi-Use-Case App\n ## Dataset: BigPatent")
        
        with gr.Tab("Patent Summarization"):
            
            summary_inputs = [gr.Textbox(lines=5, placeholder="Enter patent text"),
                              gr.Textbox(lines=2, placeholder="Enter query")]
            summary_examples = gr.Examples(qe.summary_suggestions, summary_inputs) # Query suggestions
            summary_outputs = gr.Textbox(lines=3, label="Patent Summary")
            summary_iface = gr.Interface(
                fn=analyzer.patent_summarization,
                inputs=summary_inputs,
                outputs=summary_outputs,
                flagging_mode='auto'
            )
        
        with gr.Tab("Prior Art Search"):
            prior_art_inputs = gr.Textbox(lines=2, placeholder="Enter your invention description...")
            
            prior_art_examples = gr.Examples(qe.prior_art_suggestions, prior_art_inputs) # Query suggestions
            prior_art_outputs = [gr.Textbox(lines=5, label="Potential Prior Art"),
                         gr.Textbox(label="Document Metadata")
                ]
            prior_art_iface = gr.Interface(
                fn=analyzer.prior_art_search,
                inputs=prior_art_inputs,
                outputs=prior_art_outputs,
                flagging_mode='auto'
            )
    
        with gr.Tab("Competitive Monitoring"):
            monitoring_inputs = gr.Textbox(placeholder="Enter technology area")
            monitoring_examples = gr.Examples(qe.monitoring_suggestions, monitoring_inputs) # Query suggestions
            monitoring_outputs = [gr.Textbox(lines=5, label="Competitive Landscape"),
                         gr.Textbox(label="Document Metadata")
                ]
            monitoring_iface = gr.Interface(
                fn=analyzer.competitive_monitoring,
                inputs=monitoring_inputs,
                outputs=monitoring_outputs,
                flagging_mode='auto'
            )
    
        with gr.Tab("Claims Comparison"):
            claim_inputs = [
                    gr.Textbox(lines=2, placeholder="Enter claim 1"),
                    gr.Textbox(lines=2, placeholder="Enter claim 2"),
                ]
            claim_examples = gr.Examples(qe.claim_suggestions, claim_inputs) # Query suggestions
            claim_outputs = [gr.Textbox(lines=5, label="Claim Comparison"),
                         gr.Textbox(label="Document Metadata")
                ]
            claim_iface = gr.Interface(
                fn=analyzer.claim_analysis,
                inputs=claim_inputs,
                outputs=claim_outputs,
                flagging_mode='auto'
            )  
            
        with gr.Tab("Landscape Overview"):
            landscape_inputs = gr.Textbox(placeholder="Enter CPC code")
            ladscape_examples = gr.Examples(qe.landscape_suggestions, landscape_inputs) # Query suggestions
            landscape_outputs = [gr.Textbox(lines=5, label="Landscape Overview"),
                         gr.Textbox(label="Document Metadata")
                ]
            landscape_iface = gr.Interface(
                fn=analyzer.landscape_overview,
                inputs=landscape_inputs,
                outputs=landscape_outputs,
                flagging_mode='auto'
            )
            
    demo.launch(debug=True, share=True)


if __name__ == "__main__":
    main()