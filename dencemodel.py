from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import JointAttnProcessor2_0
import torch.nn.functional as F
import torch
import pickle
import numpy as np
from PIL import Image
import gc

model_id = "stabilityai/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    cache_dir="/root/autodl-tmp/huggingface"
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16,
    cache_dir="/root/autodl-tmp/huggingface"
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.float16,
    cache_dir="/root/autodl-tmp/huggingface"
)
pipeline.enable_model_cpu_offload()



timesteps = pipeline.scheduler.timesteps
sp_sz = pipeline.transformer.config.sample_size
bsz = 4
reg_part = .3
COUNT = 0
class SD3JointAttnWithRegProcessor(JointAttnProcessor2_0):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):

        residual = hidden_states

        batch_size = hidden_states.shape[0]
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            # query torch.Size([8, 38, 4096, 64])
            # encoder_hidden_states_query_proj torch.Size([8, 38, 589, 64])
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
        #################################################################################################################
        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        global COUNT
        if COUNT/18 < 28 * reg_part:
            dtype = query.dtype
            if attn.upcast_attention:
                query = query.float()
                key = key.float()
            
            b, h, seq_len_q, d = query.shape
            _, _, seq_len_k, _ = key.shape
            query_3d = query.reshape(b*h, seq_len_q, d)
            key_3d = key.reshape(b*h, seq_len_k, d)
            value_3d = value.reshape(b*h, seq_len_k, d)
            sim = torch.baddbmm(torch.empty(query_3d.shape[0], query_3d.shape[1], key_3d.shape[1], 
                                dtype=query.dtype, device=query.device),
                    query_3d, key_3d.transpose(-1, -2), beta=0, alpha=attn.scale)

            treg = torch.pow(timesteps[COUNT//32]/1000, 5)
            min_value = sim[int(sim.size(0)/2):,:4096,:4096].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):,:4096,:4096].max(-1)[0].unsqueeze(-1)  
            mask = sreg_maps[4096].repeat(attn.heads,1,1)
            size_reg = reg_sizes[4096].repeat(attn.heads,1,1)
            sim[int(sim.size(0)/2):,:4096,:4096] += (mask>0)*size_reg*sreg*treg*(max_value-sim[int(sim.size(0)/2):,:4096,:4096])
            sim[int(sim.size(0)/2):,:4096,:4096] -= ~(mask>0)*size_reg*sreg*treg*(sim[int(sim.size(0)/2):,:4096,:4096]-min_value)

            min_value = sim[int(sim.size(0)/2):,:4096,4096:].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):,:4096,4096:].max(-1)[0].unsqueeze(-1)  
            mask = creg_maps[4096].repeat(attn.heads,1,1)
            size_reg = reg_sizes[4096].repeat(attn.heads,1,1)
            sim[int(sim.size(0)/2):,:4096,4096:] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/2):,:4096,4096:])
            sim[int(sim.size(0)/2):,:4096,4096:] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/2):,:4096,4096:]-min_value)


            attention_probs = sim.softmax(dim=-1)
            attention_probs = attention_probs.to(dtype)
            hidden_states = torch.bmm(attention_probs, value_3d)
            hidden_states = hidden_states.view(b, h, seq_len_q, d)


        else:
                # torch.Size([8, 4096, 2432])
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        COUNT += 1

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states        
            
            
    
pipeline.transformer.set_attn_processor(SD3JointAttnWithRegProcessor())

with open('./dataset/valset.pkl', 'rb') as f:
    dataset = pickle.load(f)
layout_img_root = './dataset/valset_layout/'

idx = 0
layout_img_path = layout_img_root+str(idx)+'.png'
prompts = [dataset[idx]['textual_condition']] + dataset[idx]['segment_descriptions']


text_input_1 = pipeline.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                            max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")

text_input_2 = pipeline.tokenizer_3(prompts,padding="max_length", return_length=True, return_overflowing_tokens=False, 
                            max_length=512, truncation=True,add_special_tokens=True,return_tensors="pt",)




with torch.no_grad():
    prompt_embed, pooled_prompt_embed = pipeline._get_clip_prompt_embeds(
        prompt=prompts,
        num_images_per_prompt=1,
        clip_skip=None,
        clip_model_index=0
    )

    uncond_embed, uncond_pooled_embed = pipeline._get_clip_prompt_embeds(
    prompt=[""]*bsz,
    num_images_per_prompt=1,
    clip_skip=None,
    clip_model_index=0
)
# pipeline.text_encoder.to("cpu")
torch.cuda.empty_cache()

with torch.no_grad():
    prompt_2_embed, pooled_prompt_2_embed = pipeline._get_clip_prompt_embeds(
        prompt=prompts,
        num_images_per_prompt=1,
        clip_skip=None,
        clip_model_index=1
    )
    
    uncond_2_embed, uncond_2_pooled_embed = pipeline._get_clip_prompt_embeds(
    prompt=[""]*bsz,
    num_images_per_prompt=1,
    clip_skip=None,
    clip_model_index=1
)
torch.cuda.empty_cache()


with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True):
        t5_prompt_embed = pipeline._get_t5_prompt_embeds(
            prompt=prompts, 
            num_images_per_prompt=1, 
            max_sequence_length=512
        )

    gc.collect()
    torch.cuda.empty_cache()

    with torch.cuda.amp.autocast(enabled=True):
        uncond_t5_prompt_embed = pipeline._get_t5_prompt_embeds(
            prompt=[""]*bsz, 
            num_images_per_prompt=1, 
            max_sequence_length=512
        )
torch.cuda.empty_cache()

clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
clip_prompt_embeds = torch.nn.functional.pad(
    clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
)
prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
uncond_clip_prompt_embeds = torch.cat([uncond_embed, uncond_2_embed], dim=-1)
uncond_clip_prompt_embeds = torch.nn.functional.pad(
    uncond_clip_prompt_embeds, (0, uncond_t5_prompt_embed.shape[-1] - uncond_clip_prompt_embeds.shape[-1])
)
uncond_prompt_embeds = torch.cat([uncond_clip_prompt_embeds, uncond_t5_prompt_embed], dim=-2)


layout_img_ = np.asarray(Image.open(layout_img_path).resize([sp_sz*8,sp_sz*8]))[:,:,:3]
# print("Resized layout image shape:", layout_img_.shape)
unique, counts = np.unique(np.reshape(layout_img_,(-1,3)), axis=0, return_counts=True)
sorted_idx = np.argsort(-counts)

layouts_ = []
for i in range(len(prompts)-1):
    if (unique[sorted_idx[i]] == [0, 0, 0]).all() or (unique[sorted_idx[i]] == [255, 255, 255]).all():
        layouts_ = [((layout_img_ == unique[sorted_idx[i]]).sum(-1)==3).astype(np.uint8)] + layouts_
    else:
        layouts_.append(((layout_img_ == unique[sorted_idx[i]]).sum(-1)==3).astype(np.uint8))
        
layouts = [torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).cuda() for l in layouts_]
layouts = F.interpolate(torch.cat(layouts),(sp_sz,sp_sz),mode='nearest')


###########################
###### prep for sreg ###### 
###########################
reg_part = .3
sreg = .3
creg = 1.
COUNT = 0
sreg_maps = {}
reg_sizes = {}
for r in range(4):
    res = int(sp_sz/np.power(2,r))
    layouts_s = F.interpolate(layouts,(res, res),mode='nearest')
    layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1)
    reg_sizes[np.power(res, 2)] = 1-1.*layouts_s.sum(-1, keepdim=True)/(np.power(res, 2))
    sreg_maps[np.power(res, 2)] = layouts_s


###########################
###### prep for creg ######
###########################

# print(text_input_1['length'])
# print(text_input_1['input_ids'].shape)
pww_maps1 = torch.zeros(1, 77, sp_sz, sp_sz).to("cuda:0")
for i in range(1,len(prompts)):
    wlen = text_input_1['length'][i] - 2
    widx = text_input_1['input_ids'][i][1:1+wlen]
    for j in range(77):
        if (text_input_1['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
            pww_maps1[:,j:j+wlen,:,:] = layouts[i-1:i]
            prompt_embeds[0][j:j+wlen] = prompt_embeds[i][1:1+wlen]
            print(prompts[i], i, '-th segment is handled.')
            break

print(pww_maps1.shape)
true_lengths = text_input_2['attention_mask'].sum(dim=1)
pww_maps2 = torch.zeros(1, 512, sp_sz, sp_sz).to("cuda:0")
for i in range(1,len(prompts)):
    wlen = true_lengths[i] - 1
    widx = text_input_2['input_ids'][i][:wlen]
    for j in range(512):
        if (text_input_2['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
            pww_maps2[:,j:j+wlen,:,:] = layouts[i-1:i]
            prompt_embeds[0][j:j+wlen] = prompt_embeds[i][1:1+wlen]
            print(prompts[i], i, '-th segment is handled.')
            break
print(pww_maps2.shape)
pww_maps = torch.cat([pww_maps1,pww_maps2],dim=1)
            
creg_maps = {}
for r in range(4):
    res = int(sp_sz/np.power(2,r))
    layout_c = F.interpolate(pww_maps,(res,res),mode='nearest').view(1,589,-1).permute(0,2,1).repeat(bsz,1,1)
    creg_maps[np.power(res, 2)] = layout_c

# prompt = "A painting of a couple holding a yellow umbrella in a street on a rainy night. "
image = pipeline(
    prompt=prompts[:1]*bsz,
    num_inference_steps=28,
    guidance_scale=4.5,
    max_sequence_length=512,
).images
image[0].save("image1.png")
image[1].save("image2.png")
image[2].save("image3.png")
image[3].save("image4.png")