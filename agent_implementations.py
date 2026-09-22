import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from chromadb.api import Collection
from mistralai import Mistral

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LLM utilities
class LLMUtils:
    def __init__(self, model_name: str = "mistral-small-latest"):
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name

    async def generate(self, prompt, context=None):
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        # Run blocking call in a thread
        response = await asyncio.to_thread(
            self.client.chat.complete,
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content

# Base Agent class
class BaseAgent:
    def __init__(self, llm_utils: LLMUtils):
        self.llm_utils = llm_utils

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        """Process a query and return a response"""
        raise NotImplementedError("Subclasses must implement this method")

# Router Agent
class RouterAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils):
        super().__init__(llm_utils)
        self.system_prompt = """
        You are a Router Agent for TechSolutions customer support. Your job is to:
        1. Understand the customer's query
        2. Classify the query into one of these categories:
           - Product: Questions about products, features, pricing, plans
           - Technical: Questions about errors, issues, troubleshooting
           - Billing: Questions about orders, invoices, payments, subscriptions
           - Account: Questions about user management, access, settings
           - General: General inquiries that don't fit other categories
        3. For multi-part queries, identify each part and its category
        
        Respond with JSON in this format:
        {
            "classification": "Product" or "Technical" or "Billing" or "Account" or "General",
            "confidence": 0.9, # between 0 and 1
            "requires_clarification": false, # true if query is too vague
            "clarification_question": "optional question if requires_clarification is true"
        }
        
        For multi-part queries, respond with:
        {
            "multi_part": true,
            "parts": [
                {
                    "query_part": "extracted part of the query",
                    "classification": "Product" or "Technical" or "Billing" or "Account" or "General"
                },
                ...
            ]
        }
        """

    async def process(self, query: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = f"CLASSIFY QUERY: {query}\n\nOUTPUT JSON:"
        response = await self.llm_utils.generate(prompt, self.system_prompt)
        logger.info(f"Router Agent response: {response}")
        logger.info("********************************************")
        
        # Clean the response before parsing
        cleaned_response = self._clean_json_response(response)
        logger.info(f"Cleaned response: {cleaned_response}")
        logger.info("********************************************")
        
        try:
            result = json.loads(cleaned_response)
            logger.info(f"JSON parsed successfully: {result}")
            logger.info("********************************************")
            return self._validate_response_structure(result)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed. Raw response: {response}")
            return self._safe_fallback_response(query, e)

    def _clean_json_response(self, response: str) -> str:
        """Remove non-JSON content from the response"""
        response = response.replace('``````', '')
        start = response.find('{')
        end = response.rfind('}') + 1
        return response[start:end] if start != -1 and end != 0 else response

    def _validate_response_structure(self, result: Dict) -> Dict:
        required_single = ["classification", "confidence", "requires_clarification"]
        required_multi = ["multi_part", "parts"]
        if "multi_part" in result:
            if not all(k in result for k in required_multi):
                raise ValueError("Invalid multi-part structure")
            for part in result.get("parts", []):
                if "query_part" not in part or "classification" not in part:
                    raise ValueError("Invalid part structure")
        else:
            if not all(k in result for k in required_single):
                raise ValueError("Missing required fields")
        logger.info(f"Valid response structure: {result}")
        logger.info("********************************************")       
        return result

    def _safe_fallback_response(self, query: str, error: Exception) -> Dict:
        logger.warning(f"Using fallback classification for query: {query}")
        return {
            "classification": "General",
            "confidence": 0.5,
            "requires_clarification": True,
            "clarification_question": "Could you please rephrase or provide more details about your question?",
            "parse_error": str(error)
        }

# Product Specialist Agent
class ProductSpecialistAgent(BaseAgent):
    def __init__(
        self,
        llm_utils: LLMUtils,
        product_catalog: Dict[str, Any],
        faqs: Dict[str, Any],
        vector_db: Collection,
    ):
        super().__init__(llm_utils)
        self.product_catalog = product_catalog
        self.faqs = faqs
        self.vector_db = vector_db
        self.system_prompt = """
        You are a Product Specialist Agent for TechSolutions customer support.
        You're an expert on TechSolutions products, features, pricing, and plans.
        When responding to customer queries:
        1. Be accurate and specific about product features and pricing
        2. Compare products when relevant to help customers choose
        3. Highlight benefits and use cases for specific products
        4. If you don't know something, say so rather than guessing
        Keep your responses friendly, concise, and focused on answering the customer's specific question
        Answer the Following query to the point ,just display what is asked and not any extra details.
        """

    def _retrieve_relevant_information(self, query: str) -> str:
        try:
            results = self.vector_db.query(query_texts=[query], n_results=3)
            relevant_info = "\n\n".join(results.get("documents", [[]])[0])
            return relevant_info
        except Exception as e:
            logger.error(f"Error retrieving information from vector database: {e}")
            return ""

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        relevant_info = self._retrieve_relevant_information(query)
        logger.info(f"Relevant information: {relevant_info}")
        logger.info("********************************************")
        prompt = f"""
        Customer query: {query}
        Relevant information:
        {relevant_info}
        Please provide a helpful response based on this information.
        """
        response = await self.llm_utils.generate(prompt, self.system_prompt)
        return response

# Technical Support Agent
class TechnicalSupportAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils, tech_docs: str, vector_db: Collection):
        super().__init__(llm_utils)
        self.tech_docs = tech_docs
        self.vector_db = vector_db
        self.system_prompt = """
        You are a Technical Support Agent for TechSolutions customer support.
        You're an expert in troubleshooting TechSolutions products and resolving technical issues.
        When responding to customer queries:
        1. Identify the specific issue or error described
        2. Provide step-by-step troubleshooting instructions
        3. Reference relevant documentation when applicable
        4. Suggest preventive measures for future reference
        Keep your responses clear, structured, and focused on resolving the customer's technical problem.
        """

    def _retrieve_troubleshooting_info(self, query: str) -> str:
        try:
            results = self.vector_db.query(query_texts=[query], n_results=3)
            logger.info(f"Retrieved troubleshooting information: {results}")
            logger.info("********************************************")
            relevant_info = "\n\n".join(results.get("documents", [[]])[0])
            return relevant_info
        except Exception as e:
            logger.error(f"Error retrieving troubleshooting information: {e}")
            return ""

    async def _call_diagnostic_api(self, issue_description: str) -> Dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                requests.post,
                "http://localhost:8000/api/diagnose",
                json={"description": issue_description},
            )
            response.raise_for_status()
            logger.info(f"Diagnostic API response: {response}")
            logger.info("********************************************")
            result = response.json()
            logger.info(f"Diagnostic API returned JSON: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calling diagnostic API: {e}")
            return {}

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        relevant_info = self._retrieve_troubleshooting_info(query)
        logger.info(f"Relevant troubleshooting information: {relevant_info}")
        logger.info("********************************************")
        diagnostic_info = await self._call_diagnostic_api(query)
        logger.info(f"Diagnostic API response from process function: {diagnostic_info}")
        logger.info("********************************************")
        diagnostic_text = ""
        if diagnostic_info:
            solutions = diagnostic_info.get("solutions", [])
            solutions_text = "\n".join([f"- {solution}" for solution in solutions])
            diagnostic_text = f"""
            Diagnostic results:
            Issue: {diagnostic_info.get('name', 'Unknown issue')}
            Suggested solutions:
            {solutions_text}
            Documentation: {diagnostic_info.get('documentation_link', '')}
            """
        prompt = f"""
        Customer query: {query}
        Relevant troubleshooting information:
        {relevant_info}
        {diagnostic_text}
        Please provide a helpful response to resolve this technical issue.
        """
        response = await self.llm_utils.generate(prompt, self.system_prompt)
        return response

# Order/Billing Agent
class OrderBillingAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils, product_catalog: Dict[str, Any]):
        super().__init__(llm_utils)
        self.product_catalog = product_catalog
        self.system_prompt = """
        You are an Order and Billing Agent for TechSolutions customer support.
        You're an expert in handling inquiries about orders, invoices, payments, and subscriptions.
        When responding to customer queries:
        1. Be precise about order status, payment information, and subscription details
        2. Explain billing charges clearly and transparently
        3. Outline available payment options and subscription changes when relevant
        4. Maintain a professional and reassuring tone
        Keep your responses clear, specific, and focused on addressing the customer's billing-related questions.
        """

    async def _get_order_details(self, order_id: str) -> Dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                requests.get, f"http://localhost:8000/api/orders/{order_id}"
            )
            response.raise_for_status()
            logger.info(f"Order details from the order API: {response}")
            logger.info("********************************************")
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Order not found: {order_id}")
                return {}
            else:
                logger.error(f"Error retrieving order details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving order details: {e}")
            return {}

    async def _get_account_details(self, account_id: str) -> Dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                requests.get, f"http://localhost:8000/api/accounts/{account_id}"
            )
            logger.info(f"Account details from the account API: {response}")
            logger.info("********************************************")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Account not found: {account_id}")
                return {}
            else:
                logger.error(f"Error retrieving account details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving account details: {e}")
            return {}

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        order_id = None
        import re
        order_match = re.search(r"ORD-\d+", query)
        if order_match:
            order_id = order_match.group(0)
        logger.info(f"Order ID from the process function: {order_id}")
        logger.info("********************************************")
        account_id = None
        account_match = re.search(r"ACC-\d+", query)
        if account_match:
            account_id = account_match.group(0)
        logger.info(f"Account ID from the process function: {account_id}")
        logger.info("********************************************")
        order_details = {}
        if order_id:
            order_details = await self._get_order_details(order_id)
        logger.info(f"Order details from the process function: {order_details}")
        logger.info("********************************************")
        account_details = {}
        if account_id:
            account_details = await self._get_account_details(account_id)
        logger.info(f"Account details from the process function: {account_details}")
        logger.info("********************************************")
        prompt = f"Customer query: {query}\n\n"
        if order_details:
            prompt += f"Order information:\n{json.dumps(order_details, indent=2)}\n\n"
        if account_details:
            prompt += (
                f"Account information:\n{json.dumps(account_details, indent=2)}\n\n"
            )
        if (
            "pricing" in query.lower()
            or "cost" in query.lower()
            or "price" in query.lower()
        ):
            products_info = json.dumps(
                self.product_catalog.get("products", []), indent=2
            )
            prompt += f"Product pricing information:\n{products_info}\n\n"
        prompt += "Please provide a helpful response to this billing or order question."
        response = await self.llm_utils.generate(prompt, self.system_prompt)
        logger.info(f"Billing Agent response: {response}")
        logger.info("********************************************")
        return response

# Account Management Agent
class AccountManagementAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils):
        super().__init__(llm_utils)
        self.system_prompt = """
        You are an Account Management Agent for TechSolutions customer support.
        You are an expert in handling account queries, including user management, subscription details, and available user slots.
        When responding to customer queries:
        1. Confirm the action requested (e.g., adding users).
        2. Retrieve account details to check the current subscription tier and available user slots.
        3. Provide step-by-step instructions for adding new users.
        4. Offer additional suggestions if the account has reached its user limit.
        Keep your responses clear, structured, and detailed.
        """
    
    async def _get_account_info(self, account_id: str) -> Dict[str, Any]:
        try:
            url = f"http://localhost:8000/api/accounts/{account_id}"
            response = await asyncio.to_thread(requests.get, url)
            response.raise_for_status()
            logger.info(f"Account API response: {response}")
            logger.info(f"Account API response status: {response.status_code}")
            logger.info(f"Raw account API response text: {response.text}")
            logger.info("********************************************")
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Account not found: {account_id}")
                return {}
            else:
                logger.error(f"Error retrieving account details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving account details: {e}")
            return {}
    
    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        import re
        account_id = None
        account_match = re.search(r"ACC-\d+", query)
        if account_match:
            account_id = account_match.group(0)
        if not account_id:
            account_id = "ACC-1111"
        logger.info(f"Account ID extracted: {account_id}")
        account_info = await self._get_account_info(account_id)
        logger.info(f"Retrieved account info: {account_info}")
        subscription = account_info.get("subscription", {})
        plan = subscription.get("plan", "unknown")
        if plan.lower() == "cm-pro":
            user_limit = 20
        elif plan.lower() == "cm-enterprise":
            user_limit = float("inf")
        else:
            user_limit = 5
        current_user_count = len(account_info.get("users", []))
        available_slots = "unlimited" if user_limit == float("inf") else max(0, user_limit - current_user_count)
        response_message = (
            f"I'd be happy to help you add users to your account.\n\n"
            f"Your current subscription plan is: {plan}.\n"
            f"Current number of user accounts: {current_user_count}.\n"
            f"Available user slots: {available_slots}.\n\n"
            f"To add users, please follow these steps:\n"
            f"1. Log in to the customer portal at portal.techsolutions.example.com\n"
            f"2. Navigate to Admin > User Management > Add User\n"
            f"3. Enter the email addresses for the new users\n"
            f"4. Select the appropriate role for each new user (Admin, Operator, Auditor, or Viewer)\n"
            f"5. Customize permissions if needed\n"
            f"6. Click 'Send Invitation'\n\n"
            f"If you need to add users beyond your plan's limit, you may consider purchasing additional licenses."
        )
        return response_message


    class AccountManagementAgent(BaseAgent):
        def __init__(self, llm_utils: LLMUtils):
            super().__init__(llm_utils)
            self.system_prompt = """
            You are an Account Management Agent for TechSolutions customer support.
            You are an expert in handling account queries, including subscription tiers, processing subscription change requests, and verifying account permissions.
            When responding to customer queries:
            1. Confirm the action requested (e.g., upgrade/downgrade subscription, check permissions, add users).
            2. Retrieve account details to check the current subscription tier, status, and permissions.
            3. If a subscription change is requested, simulate the change and explain the result.
            4. If a permission check is requested, verify and clearly state if the action is allowed.
            5. Provide clear, actionable next steps.
            Keep your responses clear, structured, and accurate.
            """

        async def _get_account_info(self, account_id: str) -> Dict[str, Any]:
            try:
                url = f"http://localhost:8000/api/accounts/{account_id}"
                response = await asyncio.to_thread(requests.get, url)
                response.raise_for_status()
                logger.info(f"Account API response: {response}")
                logger.info(f"Account API response status: {response.status_code}")
                logger.info(f"Raw account API response text: {response.text}")
                logger.info("********************************************")
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Account not found: {account_id}")
                    return {}
                else:
                    logger.error(f"Error retrieving account details: {e}")
                    return {}
            except Exception as e:
                logger.error(f"Error retrieving account details: {e}")
                return {}

        async def process(
            self, query: str, conversation_history: List[Dict[str, Any]] = None
        ) -> str:
            import re

            # Extract account ID from the query or use a default for testing
            account_id = None
            account_match = re.search(r"ACC-\d+", query)
            if account_match:
                account_id = account_match.group(0)
            if not account_id:
                account_id = "ACC-1111"
            logger.info(f"Account ID extracted: {account_id}")

            # Retrieve account details
            account_info = await self._get_account_info(account_id)
            logger.info(f"Retrieved account info: {account_info}")

            # Extract subscription details
            subscription = account_info.get("subscription", {})
            plan = subscription.get("plan", "unknown")
            status = subscription.get("status", "unknown")
            current_user_count = len(account_info.get("users", []))

            # Determine user limits by plan
            if plan.lower() == "cm-pro":
                user_limit = 20
            elif plan.lower() == "cm-enterprise":
                user_limit = float("inf")
            else:
                user_limit = 5

            available_slots = "unlimited" if user_limit == float("inf") else max(0, user_limit - current_user_count)

            # Identify intent: subscription tier, change request, or permission check
            query_lower = query.lower()
            response_message = ""

            # 1. Handle subscription tier questions
            if "subscription" in query_lower or "plan" in query_lower or "tier" in query_lower:
                response_message += (
                    f"Your current subscription plan is: **{plan}** (status: {status}).\n"
                    f"You have {current_user_count} user(s) and {available_slots} user slots available.\n"
                )

            # 2. Handle subscription change requests (upgrade/downgrade)
            if "upgrade" in query_lower or "downgrade" in query_lower or "change plan" in query_lower:
                if "upgrade" in query_lower or ("change" in query_lower and "enterprise" in query_lower):
                    if plan.lower() == "cm-enterprise":
                        response_message += "You are already on the highest (Enterprise) plan.\n"
                    else:
                        response_message += (
                            "Subscription change request detected: **Upgrade to Enterprise**.\n"
                            "Request processed successfully. Your account will be upgraded to the Enterprise plan on your next billing cycle.\n"
                            "You will now have unlimited user slots and access to all premium features.\n"
                        )
                elif "downgrade" in query_lower or ("change" in query_lower and "pro" in query_lower):
                    if plan.lower() == "cm-pro":
                        response_message += "You are already on the Pro plan.\n"
                    else:
                        response_message += (
                            "Subscription change request detected: **Downgrade to Pro**.\n"
                            "Request processed successfully. Your account will be downgraded to the Pro plan on your next billing cycle.\n"
                            "User slots will be limited to 20 after the downgrade.\n"
                        )
                else:
                    response_message += (
                        "Subscription change request detected. Please specify the plan you want to switch to (Pro or Enterprise).\n"
                    )

            # 3. Handle permission checks
            if "permission" in query_lower or "allowed" in query_lower or "can i" in query_lower:
                # Example: "Can I add more users?" or "Do I have permission to manage billing?"
                if "add user" in query_lower or "add more user" in query_lower:
                    if user_limit == float("inf") or current_user_count < user_limit:
                        response_message += (
                            f"You have permission to add more users. "
                            f"Current users: {current_user_count}, available slots: {available_slots}.\n"
                        )
                    else:
                        response_message += (
                            f"You have reached your user limit ({user_limit}). "
                            "Please upgrade your subscription or remove existing users to add more.\n"
                        )
                elif "manage billing" in query_lower or "billing" in query_lower:
                    roles = [user.get("role", "") for user in account_info.get("users", [])]
                    if any(role in ["admin", "billing"] for role in roles):
                        response_message += "You have permission to manage billing for this account.\n"
                    else:
                        response_message += "You do not have billing permissions. Please contact your account admin.\n"
                else:
                    response_message += (
                        "Please specify the action you want to check permissions for (e.g., add users, manage billing).\n"
                    )

            # If no specific intent detected, provide a general account summary
            if not response_message:
                response_message = (
                    f"Account: {account_info.get('name', 'Unknown')}\n"
                    f"Plan: {plan}, Status: {status}\n"
                    f"Users: {current_user_count}, Available slots: {available_slots}\n"
                    "Let me know if you want to change your subscription or check permissions for a specific action.\n"
                )

            return response_message


# Orchestrator implementation
class AgentOrchestrator:
    def __init__(
        self,
        llm_utils: LLMUtils,
        knowledge_base: Dict[str, Any],
        vector_db: Dict[str, Collection],
    ):
        self.llm_utils = llm_utils
        self.knowledge_base = knowledge_base
        self.vector_db = vector_db
        self.conversations = {}
        self.router_agent = RouterAgent(llm_utils)
        self.product_agent = ProductSpecialistAgent(
            llm_utils,
            knowledge_base["product_catalog"],
            knowledge_base["faqs"],
            vector_db["products"],
        )
        self.technical_agent = TechnicalSupportAgent(
            llm_utils, knowledge_base["tech_docs"], vector_db["technical"]
        )
        self.billing_agent = OrderBillingAgent(
            llm_utils, knowledge_base["product_catalog"]
        )
        self.account_agent = AccountManagementAgent(llm_utils)
        logger.info("Agent Orchestrator initialized with all agents")

    async def process_query(
        self, query: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        conversation_history = self.conversations.get(conversation_id, [])
        routing_result = await self.router_agent.process(query, conversation_history)
        if routing_result.get("multi_part", False):
            responses = []
            for part in routing_result.get("parts", []):
                part_query = part.get("query_part")
                part_classification = part.get("classification")
                part_response = await self._process_single_query(
                    part_query, part_classification, conversation_history
                )
                responses.append(f"{part_response}")
            final_response = "\n\n".join(responses)
            agent_type = "multiple"
        else:
            classification = routing_result.get("classification", "General")
            if routing_result.get("requires_clarification", False):
                final_response = routing_result.get(
                    "clarification_question",
                    "Could you please provide more details about your question?",
                )
                agent_type = "router"
            else:
                final_response = await self._process_single_query(
                    query, classification, conversation_history
                )
                agent_type = classification.lower()
        self.conversations[conversation_id].append(
            {"query": query, "response": final_response, "agent": agent_type}
        )
        return {
            "response": final_response,
            "agent": agent_type,
            "conversation_id": conversation_id or "new_conversation",
        }

    async def _process_single_query(
        self,
        query: str,
        classification: str,
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        if classification == "Product":
            logger.info(f"Processing product query, query: {query}")
            logger.info("********************************************")
            return await self.product_agent.process(query, conversation_history)
        elif classification == "Technical":
            logger.info(f"Processing technical query, query: {query}")
            logger.info("********************************************")
            return await self.technical_agent.process(query, conversation_history)
        elif classification == "Billing":
            logger.info(f"Processing billing query, query: {query}")
            logger.info("********************************************")
            return await self.billing_agent.process(query, conversation_history)
        elif classification == "Account":
            if self.account_agent:
                logger.info(f"Processing account query, query: {query}")
                logger.info("********************************************")
                return await self.account_agent.process(query, conversation_history)
            else:
                fallback_response = await self.billing_agent.process(
                    query, conversation_history
                )
                return fallback_response
        else:
            general_prompt = f"""
            Customer query: {query}
            Please provide a helpful and friendly general response to this query.
            """
            general_system_prompt = """
            You are a Customer Support Agent for TechSolutions.
            Provide helpful, friendly, and concise responses to general customer inquiries.
            If the query should be handled by a specialist agent, indicate which type of specialist would be appropriate.
            """
            return await self.llm_utils.generate(
                general_prompt, general_system_prompt
            )

    def add_account_management_agent(self):
        """Add the Account Management Agent (for mid-session challenge)"""
        pass
