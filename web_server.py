from flask import Flask, Response, request
from agent import Agent
import uuid
import config
import utils
import json
app = Flask(__name__)

agents = {}

@app.route('/')
def hello_world():
    return 'Flask Dockerized'

@app.route('/new_agent/<workspaces>')
def new_agent(workspaces):
    agent = Agent().build_agent(workspaces)
    key = uuid.uuid4().hex[:16].upper()
    agents[key] = agent
    return key

@app.route('/choose_workspace/<agent_id>')
def choose_workspace(agent_id):
    if agent_id in agents:
        action = agents[agent_id].act()
        data = {'action': action}
        response = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json')
        return response
    else:
        return Response(status=400)

@app.route('/get_reward', methods = ['POST'])
def get_reward():
    data = request.get_json()
    key = data.get("key")
    action = data.get("action")
    reward = data.get("reward")
    if key is None or action is None or reward is None:
        return Response(status=400)
    elif key not in agents:
        return Response(status=400)
    else:
        agent = agents[key]
        agent.aggregate_observation(action, reward)
        if agent.get_memory_size() is config.training_size:
            for i in range(config.nb_batches):
                action_batch, reward_batch = utils.shuffle_batch(agent.get_memory_actions(), 
                    agent.get_memory_rewards())
                loss_value,upd,resp,ww = agent.train(action_batch, reward_batch)
            agent.clear_memory()
            response = app.response_class(
                response=json.dumps({'training_size': str(config.training_size), 
                    'loss': str(loss_value), 'weights': str(ww)}),
                status=200,
                mimetype='application/json')
            return response
        else:
            data = {'training_size': agent.get_memory_size()}
            response = app.response_class(
                response=json.dumps(data),
                status=200,
                mimetype='application/json')
        return response


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')