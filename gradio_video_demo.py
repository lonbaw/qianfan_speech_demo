from langchain.chat_models import ErnieBotChat
import os, uuid
import gradio as gr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import tempfile
from aip import AipSpeech
from pydub import AudioSegment
from pydub.playback import play
# tempfile.tempdir = "./tmp"

llm = ErnieBotChat(ernie_client_id="xxxx", 
            ernie_client_secret="xxxxxx",
            model_name='ERNIE-Bot',
            temperature=0.01
    ) 
template = """You are a chatbot having a conversation with a human. Please answer as briefly as possible.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

memory = ConversationBufferMemory(llm=llm,memory_key="chat_history",return_messages=True)
conversation = LLMChain(llm=llm, memory=memory,prompt=prompt)


APP_ID = 'xxxx'
API_KEY = 'xxxx'
SECRET_KEY = 'xxxxxx'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def transcribe(audio):
    with tempfile.TemporaryDirectory() as tempdir:
        print(f"Temporary directory created at {tempdir}")
        audio = AudioSegment.from_file(audio)

        new_frame_rate = 16000
        new_audio = audio.set_frame_rate(new_frame_rate)

        random_code = uuid.uuid4()
        temp_audio_path = f"{tempdir}/output_audio_16000_{random_code}.wav"
        new_audio.export(temp_audio_path, format="wav")

                
        def get_file_content(filePath):
            with open(filePath, 'rb') as fp:
                return fp.read()

        ret = client.asr(get_file_content(temp_audio_path), 'wav', 16000, {'dev_pid': 1537})
        # print(f"check2, done: {ret}")

    # print(f"test1:{ret}, test2:{type(ret)}")
    # 注意：退出 with 块后，tempdir 及其内容会被自动删除
    return ret.get('result')[0]


def play_voice(text):
    result = client.synthesis(text, 'zh', 1, {'vol': 5})
    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3', mode='wb') as temp_audio_file:
            temp_audio_file.write(result)
            temp_audio_file.seek(0)  # 回到文件开头
            # print(temp_audio_file.name)
            # 使用 pydub 播放音频
            audio = AudioSegment.from_file(temp_audio_file.name, format="mp3")
            play(audio)

# 录音文件转文本的过程
def process_audio(audio, history=[]):
    if audio:
        text = transcribe(audio)  
        # print(text)
        if text is None:
            text="你好"
        responses, clear, updated_history = predict(text, history)
        # print(f"====={responses}")
        # print(f"++++{updated_history}")
        # 返回处理结果和 None 来清空音频输入
        return responses, None, updated_history    
    else:       
        # print(f"------{history}")
        return [(u,b) for u,b in zip(history[::2], history[1::2])], None, history
        
    
# 调用openai对话功能
def predict(input, history=[]):    
    history.append(input)    
    response = conversation.predict(human_input=input)
    history.append(response)
    # history[::2] 切片语法，每隔两个元素提取一个元素，即提取出所有的输入，
    # history[1::2]表示从历史记录中每隔2个元素提取一个元素，即提取出所有的输出
    # zip函数把两个列表元素打包为元组的列表的方式

    play_voice(response)
    responses = [(u,b) for u,b in zip(history[::2], history[1::2])]
    print("==取出输入：",history[::2])
    print("==取出输出：",history[1::2])
    print("组合元组：",responses)
    
    return responses, "", history
 
def clear_history():
    # 返回一个空列表来清空聊天记录
    return [], []

with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
 
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
    # 录音功能
    with gr.Row(): 
        # 得到音频文件地址
        audio = gr.Audio(sources="microphone", type="filepath")
    with gr.Row():        
        clear_button = gr.Button("清空聊天记录")
    # reset_button = gr.Button("重置录音")    
    txt.submit(predict, [txt, state], [chatbot, txt, state])    
    audio.change(process_audio, [audio, state], [chatbot, audio, state])

    clear_button.click(clear_history, [], [chatbot, state])            
# 启动gradio
demo.launch(share=False)