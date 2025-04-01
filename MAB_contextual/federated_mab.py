import numpy as np

class FederatedAggregator:
    """
    A federated learning aggregator for Multi-Agent MAB.
    UAVs, Vehicles, and Base Stations send updates to this class,local_q_value
    which aggregates Q-values and sends back the updated global values.
    """

    def __init__(self):
        self.global_q_values = {}  # Stores aggregated Q-values for all nodes

    def receive_local_update(self, entity_id, local_q_values):
        """
        Receives Q-value updates from UAVs, Vehicles, and BSs.

        Parameters:
            entity_id (str): Identifier for the agent (UAV/Vehicle/BS).
            local_q_values (dict): The agent’s Q-values (structured as a nested dictionary).

        Updates the global_q_values dictionary using a federated averaging approach.
        """

        if entity_id not in self.global_q_values:
            self.global_q_values[entity_id] = {}  # Initialize for this entity

        for content_type, content_coords in local_q_values.items():
            if content_type not in self.global_q_values[entity_id]:
                self.global_q_values[entity_id][content_type] = {}

            for coord, categories in content_coords.items():
                if coord not in self.global_q_values[entity_id][content_type]:
                    self.global_q_values[entity_id][content_type][coord] = {}

                for category, contents in categories.items():
                    if category not in self.global_q_values[entity_id][content_type][coord]:
                        self.global_q_values[entity_id][content_type][coord][category] = {}

                    for content_no, values in contents.items():
                        if content_no not in self.global_q_values[entity_id][content_type][coord][category]:
                            # First entry: Only store `q_value`
                            self.global_q_values[entity_id][content_type][coord][category][
                                content_no] = self.global_q_values[entity_id][content_type][coord][category][content_no] = {
                            'q_value': values.get('q_value', 0)  # Default to 0 if missing
                        }
                        else:
                            # Merge Q-values using federated averaging
                            old_q = self.global_q_values[entity_id][content_type][coord][category][content_no][
                                'q_value']
                            new_q = values.get('q_value', 0)

                            α = 0.5  # Weight for federated averaging
                            self.global_q_values[entity_id][content_type][coord][category][content_no]['q_value'] = (
                                    α * old_q + (1 - α) * new_q
                            )

        print(f"Updated Global Q-Values from {entity_id}")

    import numpy as np

    def aggregate_updates(self):
        """
        Aggregates Q-values across all entities using Federated Averaging.
        Each unique content item is identified by [content_type][coord][category][content_no]
        and its Q-value is averaged across all entities that contributed to it.
        """

        aggregated_q_values = {}  # Stores aggregated Q-values

        # Step 1: Collect all Q-values for each unique content
        for entity_id, q_table in self.global_q_values.items():
            for content_type, content_coords in q_table.items():
                if content_type not in aggregated_q_values:
                    aggregated_q_values[content_type] = {}

                for coord, categories in content_coords.items():
                    if coord not in aggregated_q_values[content_type]:
                        aggregated_q_values[content_type][coord] = {}

                    for category, contents in categories.items():
                        if category not in aggregated_q_values[content_type][coord]:
                            aggregated_q_values[content_type][coord][category] = {}

                        for content_no, values in contents.items():
                            if content_no not in aggregated_q_values[content_type][coord][category]:
                                aggregated_q_values[content_type][coord][category][content_no] = {
                                    'q_values': [],  # List to store Q-values from different entities
                                }

                            # Store Q-value from this entity
                            q_value = values['q_value']
                            aggregated_q_values[content_type][coord][category][content_no]['q_values'].append(q_value)

        # Step 2: Compute the federated average for each unique content
        for content_type in aggregated_q_values:
            for coord in aggregated_q_values[content_type]:
                for category in aggregated_q_values[content_type][coord]:
                    for content_no in aggregated_q_values[content_type][coord][category]:
                        q_list = aggregated_q_values[content_type][coord][category][content_no]['q_values']
                        if q_list:  # Ensure there are values to aggregate
                            aggregated_q_values[content_type][coord][category][content_no]['q_value'] = np.mean(q_list)

        # Step 3: Update the global Q-values with the aggregated results
        #self.global_q_values = aggregated_q_values  # Replace with aggregated values

        print("Global Q-values aggregated successfully.")

    def get_federated_q_value(self, content):
        """
        Retrieve the federated (aggregated) Q-value for a given content.

        Parameters:
            content (dict): Dictionary containing content metadata
                            (content_type, content_coord, content_category, content_no).

        Returns:
            float: Federated Q-value if found, otherwise returns 0.
        """

        # Extract content details
        content_type = content.get('content_type')
        coord = content.get('content_coord')
        category = content.get('content_category')
        content_no = content.get('content_no')

        # Ensure the aggregated_q_values structure exists
        if not hasattr(self, 'aggregated_q_values'):
            self.aggregated_q_values = {}

        # Check if the content exists in the aggregated Q-values
        if (content_type in self.aggregated_q_values and
                coord in self.aggregated_q_values[content_type] and
                category in self.aggregated_q_values[content_type][coord] and
                content_no in self.aggregated_q_values[content_type][coord][category]):
            return self.aggregated_q_values[content_type][coord][category][content_no]['q_value']

        # Return default value if content is not found
        return 0


