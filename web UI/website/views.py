from flask import Blueprint, render_template, flash, request, jsonify
from flask_login import login_required, current_user
from .models import Note 
from . import db 
import json 
from BasicChatbot.Chatbot import respond


views = Blueprint("views", __name__)
@views.route("/home", methods=['GET', 'POST'])
@login_required
def home():

    if request.method == 'POST': 
        note = request.form.get('note')#Gets the note from the HTML 
        
        if len(note) < 1:
            flash('Note is too short!', category='error') 
        else:
            resp, tag = respond(note)

            new_note = Note(data=note, response = resp, user_id=current_user.id)  #providing the schema for the note 
            db.session.add(new_note) #adding the note to the database 
            db.session.commit()
    return render_template("home.html", user = current_user)

@views.route('/delete-note', methods=['POST'])
def delete_note():  
    note = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})