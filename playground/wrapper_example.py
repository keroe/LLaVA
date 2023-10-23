from llava.model.llava_wrapper import LlavaWrapper

if __name__ == "__main__":
	llava_wrapper = LlavaWrapper("liuhaotian/llava-v1.5-7b")
	image_file ="https://llava-vl.github.io/static/images/view.jpg"
	query = "What can you see in the image?"
	output = llava_wrapper.run(query, image_file)