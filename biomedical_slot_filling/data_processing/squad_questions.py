import json
from json import JSONEncoder

import attr

class MyJsonEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

class QuestionAnswers:
    id = attr.attrib()
    question = attr.attrib()
    answers = attr.attrib()

    def __str__(self):
        return self.id+' '+self.question

class SquadQuestion:
    qas = attr.attrib()
    context = attr.attrib()
    docid = attr.attrib()

    def __str__(self):
        return ' '.join([str(qa) for qa in self.qas])+' '+self.context


def write_to_json(questions, json_file, version):
    squad_json = {}
    squad_json['version'] = version
    squad_json['data'] = []
    paragraphs = {}
    paragraphs['paragraphs'] = []
    for question in questions:
        paragraphs['paragraphs'].append(
            {
                'context':question.context,
                'docid':question.docid,
                'qas': question.qas
            }
        )
    squad_json['data'].append(paragraphs)
    with open(json_file, 'w') as fw:
        json.dump(squad_json, fw, indent=1, cls=MyJsonEncoder)

# write_qa_test_set()
def build_sq(question, result, docid, i):
    qa = QuestionAnswers()
    qa.question = question
    qa.answers = []
    qa.id = str(i)
    qas = [qa]
    sq = SquadQuestion()
    sq.qas = qas
    sq.context = result
    sq.docid = docid
    return sq


def write_questions_and_context_in_squad_json(questions, results, output_file):
    sqs = []
    for question in questions:
        for i, (docid, result) in enumerate(results[question]):
            sq = build_sq(question=question, result=result, docid=docid, i=docid+'_'+str(i))
            sqs.append(sq)
    write_to_json(questions=sqs, json_file=output_file, version='BioASQ')