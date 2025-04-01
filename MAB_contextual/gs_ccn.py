
import random
import time

class GroundStation:
    def __init__(self, gs_id):
        self.gs_id = gs_id
        self.content_cache = []  # List to store cached content
        self.content_index = 0  # Index to keep track of the content
        self.content_categories = ['I', 'II', 'III', 'IV']
        self.content_cache = { }  # Dictionary to store content category-wise

    def cache_clean(self, current_time, satellites):
        for satellite_id in satellites:
            if satellite_id in self.content_cache:
                for category in self.content_categories:
                    if category in self.content_cache[satellite_id]:
                        cached_content = self.content_cache[satellite_id][category]
                        # Filter out content that has expired
                        self.content_cache[satellite_id][category] = [content for content in cached_content if
                                                                                current_time < content[
                                                                                    'generation_time'] + content[
                                                        'content_validity']]

    def find_content_in_gs(self, content_request):
        requested_entity_type = content_request['type']
        requested_entity_no = content_request['coord']
        requested_category = content_request['category']
        requested_content_no = content_request['no']

        flag = 0
        if requested_entity_type == 'satellite':
            # Check if the requested content is in the content cache for the specified category
            if requested_entity_no in self.content_cache:
                if requested_category in self.content_cache[requested_entity_no]:
                    category_cache = self.content_cache[requested_entity_no][requested_category]
                    for content in category_cache:
                        if content['content_no'] == content_request['no']:
                            #print('from gs')
                            flag = 1
                            return content

        if flag == 0:
            return None


    def add_to_cache(self, satellite_id, category, contents):
        if satellite_id not in self.content_cache:
            self.content_cache[satellite_id] = {}
        if category not in self.content_cache[satellite_id]:
             self.content_cache[satellite_id][category] = {}
        self.content_cache[satellite_id][category]=contents

    def print_cache(self):
        print("Cache for GS")
        for satellite_id, categories in self.content_cache.items():
            print(f"Category: {satellite_id}", end=' ')
            for category, contents in categories.items():
                print(f"Category: {category}", end=' ')
                for content in contents:
                    print(f"Content Number: {content['content_no']}", end=' ')
                print()  # Print a new line

    def receive_request(self, content_request):
        # Function to search for content in the cache based on the requested content_request
        requested_content = []
        for content in self.content_cache:
            if (content['element_type'] == content_request['element_type'] and
                content['content_type'] == content_request['content_type'] and
                content['content_no'] == content_request['no']):
                requested_content.append(content)
        self.cache_lock.release()
        return requested_content

    def run(self, current_time, satellites):
        self.cache_clean(current_time, satellites)
        #self.print_cache()

