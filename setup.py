import discord, torch
import interactions

from concurrent.futures import ThreadPoolExecutor
import asyncio
from transformers import LlamaTokenizer, LlamaForCausalLM


bot = interactions.Client(...)



@bot.event
async def on_start():
    print(f"Logged in as {self.bot.user}")
    asyncio.get_running_loop().create_task(background_task())


@client.event
async def on_message(message):
    if message.author == self.bot.user:
        return

    if isinstance(message.channel, discord.channel.DMChannel) or (
        self.bot.user and self.bot.user.mentioned_in(message)
    ):
        if message.reference:
            pastMessage = await message.channel.fetch_message(
                message.reference.message_id
            )
        else:
            pastMessage = None
        await queue.put((message, pastMessage))


@self.slash.slash(
    name="fetch_chat_history",
    description="Fetches and analyzes the last n messages from the chat history.",
    options=[
        create_option(
            name="num_messages",
            description="Number of past messages to fetch",
            option_type=4,
            required=True,
        )
    ],
)
async def fetch_chat_history(self, ctx: SlashContext, num_messages: int):
    messages = await ctx.channel.history(limit=num_messages).flatten()
    chat_history = " ".join([msg.content for msg in messages])

    # Tokenize and run the model (replace this part with your actual model's code)
    input_data = self.tokenizer(chat_history, return_tensors="pt")
    output = self.model.generate(**input_data)

    # Decode and send the response (replace this part with your actual model's code)
    response = self.tokenizer.decode(output[0])
    await ctx.send(f"Model Response: {response}")


def sync_task(message):
    input_ids = tokenizer(message, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=250,
        do_sample=True,
        repetition_penalty=1.3,
        temperature=0.8,
        top_p=0.75,
        top_k=40,
    )
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1] :])
    return response


async def background_task():
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    print("Task Started. Waiting for inputs.")
    while True:
        msg_pair: tuple[discord.Message, discord.Message] = await queue.get()
        msg, past = msg_pair

        username = self.bot.user.name
        user_id = self.bot.user.id
        message_content = msg.content.replace(f"@{username} ", "").replace(
            f"<@{user_id}> ", ""
        )
        past_content = None
        if past:
            past_content = past.content.replace(f"@{username} ", "").replace(
                f"<@{user_id}> ", ""
            )
        text = generate_prompt(message_content, past_content)
        response = await loop.run_in_executor(executor, sync_task, text)
        print(f"Response: {text}\n{response}")
        await msg.reply(response, mention_author=False)


def generate_prompt(text, pastMessage):
    if pastMessage:
        return f"""### Instruction:
Your previous response to the prior instruction: {pastMessage}
        
Current instruction to respond to: {text}
### Response:"""
    else:
        return f"""### Instruction:
{text}
### Response:"""


if __name__ == "__main__":
    bot = interactions.Client(token="your_secret_bot_token")

    alpaca_bot = AlpacaBot(bot)
    bot.add_cog(alpaca_bot)

    bot.start
